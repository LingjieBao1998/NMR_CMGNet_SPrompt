from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse, FileResponse
from django.contrib.auth.models import User
from django.core.paginator import Paginator
from django.db.models import Q, Count, Avg
from django.utils import timezone
from datetime import datetime, timedelta
import json
import os
import logging

logger = logging.getLogger(__name__)

from .models import NMRTask, TaskLog, ContactMessage, SystemStatistics
from .tasks import process_nmr_task
from .crypto_utils import generate_task_token
from .email_utils import send_task_confirmation_email, send_task_result_email, send_task_failed_email


def get_user_task_or_404(request, task_id):
    """获取属于当前登录用户的任务，否则 404（数据隔离核心）

    超级管理员可访问所有任务，便于运维排查。
    """
    queryset = NMRTask.objects.all()
    if not request.user.is_superuser:
        queryset = queryset.filter(submitted_by=request.user)
    return get_object_or_404(queryset, task_id=task_id)


def count_valid_nmr_from_file(uploaded_file):
    """
    Count valid NMR entries from uploaded CSV file.

    Args:
        uploaded_file: Django uploaded file object

    Returns:
        int: Number of valid NMR entries
    """
    import csv
    import io

    try:
        # Read the uploaded file
        csv_file = io.TextIOWrapper(uploaded_file.file, encoding='utf-8')
        csv_reader = csv.reader(csv_file)

        # Skip header row
        next(csv_reader, None)

        valid_count = 0
        for row in csv_reader:
            if len(row) >= 2:  # Ensure row has at least 2 columns
                nmr_value = row[1].strip()  # NMR column is typically the second column
                if nmr_value:  # Check if NMR value is not empty
                    valid_count += 1

        # Reset file pointer for later reading
        uploaded_file.file.seek(0)

        return valid_count

    except Exception as e:
        logger.error(f"Error counting NMR entries: {str(e)}")
        return 1  # Default to 1 if counting fails


def process_uploaded_csv_file(uploaded_file, dest_path=None):
    """
    Process uploaded CSV file and optionally save valid NMR entries to ori.csv

    Args:
        uploaded_file: Django uploaded file object
        dest_path: Path to save the processed CSV file (optional, if None then only process and return data)

    Returns:
        list: List of valid NMR data rows [id, nmr_value]
    """
    import csv
    import io

    valid_nmr_data = []

    try:
        # Read the uploaded file
        csv_file = io.TextIOWrapper(uploaded_file.file, encoding='utf-8')
        csv_reader = csv.reader(csv_file)

        # Read header row
        header = next(csv_reader, None)

        # Process data rows
        row_id = 1
        for row in csv_reader:
            if len(row) >= 2:  # Ensure row has at least 2 columns
                nmr_value = row[1].strip()  # NMR column is typically the second column
                if nmr_value:  # Check if NMR value is not empty
                    valid_nmr_data.append([row_id, nmr_value])
                    row_id += 1

        # Only write to file if dest_path is provided
        if dest_path:
            # Write valid data to ori.csv
            with open(dest_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'NMR'])  # Header
                writer.writerows(valid_nmr_data)  # Data rows

            logger.info(f"Processed {len(valid_nmr_data)} valid NMR entries to {dest_path}")
        else:
            logger.info(f"Processed {len(valid_nmr_data)} valid NMR entries (no file saved)")

        # Reset file pointer for potential re-reading
        uploaded_file.file.seek(0)

        return valid_nmr_data

    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        # Fallback: return minimal data
        return [[1, ""]]


@login_required
def home(request):
    """Home page view"""
    # Get system statistics
    today = timezone.now().date()
    
    # Recent tasks (仅当前用户的)
    recent_tasks = NMRTask.objects.select_related('submitted_by').filter(
        submitted_by=request.user
    ).order_by('-submitted_at')[:10]
    
    # Today's statistics
    today_stats = SystemStatistics.objects.filter(date=today).first()
    if not today_stats:
        # Calculate on-the-fly if not exists
        today_tasks = NMRTask.objects.filter(submitted_at__date=today)
        today_stats = {
            'total_tasks': today_tasks.count(),
            'completed_tasks': today_tasks.filter(status='completed').count(),
            'failed_tasks': today_tasks.filter(status='failed').count(),
            'active_users': today_tasks.values('submitted_by').distinct().count()
        }
    
    # Queue statistics (仅当前用户自己的任务，与队列页/API一致)
    user_tasks = NMRTask.objects.filter(submitted_by=request.user)
    queue_stats = {
        'all': user_tasks.count(),
        'pending': user_tasks.filter(status='pending').count(),
        'processing': user_tasks.filter(status='processing').count(),
        'completed': user_tasks.filter(status='completed').count(),
        'failed': user_tasks.filter(status='failed').count(),
    }

    context = {
        'page_title': 'NMR Core Detection System',
        'recent_tasks': recent_tasks,
        'today_stats': today_stats,
        'queue_stats': queue_stats,
    }
    return render(request, 'core_detection/home.html', context)


@login_required
def submit_task(request):
    """Submit new task page"""
    if request.method == 'POST':
        try:
            # Get form data
            task_name = request.POST.get('task_name')
            task_type = request.POST.get('task_type', 'single_molecule')
            input_nmr = request.POST.get('input_nmr', '')
            email = request.POST.get('email', '')
            user_name = request.POST.get('user_name', '')
            
            # Get advanced options
            model_selection = request.POST.get('model_selection', 'ensemble')
            xgboost_threshold = request.POST.get('xgboost_threshold', '0.5')
            topk = request.POST.get('topk', '10')
            
            # Validate model selection
            if model_selection not in ['ensemble', 'xgboost']:
                model_selection = 'ensemble'  # Default to ensemble
            
            # Validate and clip XGBoost threshold
            try:
                threshold_value = float(xgboost_threshold)
                # Clip to range [0.1, 0.999]
                xgboost_threshold = str(max(0.1, min(0.999, threshold_value)))
            except ValueError:
                xgboost_threshold = '0.5'  # Default value
            
            # Validate and clip topk predictions (only for ensemble model)
            try:
                topk_number = int(topk)
                # Clip to valid range [1, 30]
                topk_number = max(1, min(30, topk_number))
                topk = topk_number
            except (ValueError, TypeError):
                topk = 10  # Default value
            
            # For XGBoost model, set topk to minimal
            if model_selection == 'xgboost':
                topk = 1  # Minimal predictions for XGBoost only

            # Handle file upload if present
            input_file = request.FILES.get('input_file')

            # Validate input: exactly one of input_nmr or input_file must be provided
            has_input_nmr = bool(input_nmr.strip())
            has_input_file = bool(input_file)

            if has_input_nmr and has_input_file:
                return JsonResponse({'error': 'Please provide either NMR data text OR upload a file, not both.'}, status=400)
            elif not has_input_nmr and not has_input_file:
                return JsonResponse({'error': 'Please provide either NMR data text OR upload a file.'}, status=400)
            
            # Build parameters dict with all options
            parameters = {
                'model_selection': model_selection,
                'xgboost_threshold': float(xgboost_threshold),
                'topk': topk,
            }

            # Calculate total_count based on input type and validate file content
            total_count = 1  # Default for input_nmr
            valid_nmr_data = []  # For storing processed CSV data

            if input_file and not has_input_nmr:
                # For file uploads, we need to read the CSV to count valid NMR rows
                try:
                    # Read the file content once and store it
                    import csv
                    import io

                    # Read file content
                    file_content = input_file.read()
                    input_file.seek(0)  # Reset file pointer

                    # Parse CSV content
                    csv_file = io.TextIOWrapper(io.BytesIO(file_content), encoding='utf-8')
                    csv_reader = csv.reader(csv_file)

                    # Skip header
                    next(csv_reader, None)

                    # Count valid rows and store data
                    valid_count = 0
                    temp_valid_data = []
                    row_id = 1

                    for row in csv_reader:
                        if len(row) >= 2:
                            nmr_value = row[1].strip()
                            if nmr_value:
                                valid_count += 1
                                temp_valid_data.append([row_id, nmr_value])
                                row_id += 1

                    total_count = valid_count
                    valid_nmr_data = temp_valid_data

                    logger.info(f"Counted {total_count} valid NMR entries from uploaded file")

                    # Check if there are any valid NMR entries
                    if total_count == 0:
                        return JsonResponse({
                            'error': 'The uploaded CSV file does not contain any valid NMR data. '
                                   'Please ensure your file has at least one row with NMR data in the second column.'
                        }, status=400)

                    logger.info(f"Validated {len(valid_nmr_data)} NMR entries from CSV")

                except Exception as e:
                    logger.error(f"Error processing NMR entries from file: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return JsonResponse({
                        'error': f'Error reading uploaded CSV file: {str(e)}'
                    }, status=400)

            # 任务归属当前登录用户（数据隔离）
            # Create new task
            task = NMRTask.objects.create(
                task_name=task_name,
                task_type=task_type,
                submitted_by=request.user,
                input_nmr=input_nmr,
                input_file=input_file,
                parameters=parameters,
                email=email,
                user_name=user_name,
                status='pending',
                total_count=total_count
            )
            
            # Generate task token
            task_token = generate_task_token(task.task_id)
            task.task_token = task_token
            task.save()

            # Generate celery_task_id for file paths
            import uuid
            task.celery_task_id = str(uuid.uuid4())
            task.save()

            logger.info(f"Task {task.task_id} with celery_task_id: {task.celery_task_id}")

            # 添加调试日志
            logger.info(f"CSV Creation Debug - input_nmr: '{input_nmr}', input_file: {input_file}")
            logger.info(f"CSV Creation Debug - input_nmr bool: {bool(input_nmr)}, not input_file: {not input_file}")

            # Save input_nmr as CSV file if provided (BEFORE submitting to Celery)
            if input_nmr and not input_file:
                try:
                    from pathlib import Path
                    import csv

                    # Create result directory
                    project_dir = Path(__file__).parent.parent
                    result_dir = project_dir / "result" / task.celery_task_id
                    result_dir.mkdir(parents=True, exist_ok=True)

                    # Save as CSV file (format: id,NMR)
                    csv_file_path = result_dir / "ori.csv"
                    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['id', 'NMR'])  # Header
                        writer.writerow([1, input_nmr])  # Data

                    logger.info(f"Saved input_nmr to CSV file: {csv_file_path}")

                except Exception as e:
                    logger.error(f"Error saving input_nmr to CSV: {str(e)}")

            # Save processed CSV file to result directory if provided
            logger.info(f"CSV file save check - input_file: {bool(input_file)}, valid_nmr_data: {len(valid_nmr_data) if valid_nmr_data else 0}")
            if input_file and valid_nmr_data:
                try:
                    from pathlib import Path
                    import csv

                    # Create result directory
                    project_dir = Path(__file__).parent.parent
                    result_dir = project_dir / "result" / task.celery_task_id
                    logger.info(f"Creating directory: {result_dir}")
                    result_dir.mkdir(parents=True, exist_ok=True)

                    dest_path = result_dir / "ori.csv"
                    logger.info(f"Target CSV path: {dest_path}")

                    # Write the already processed data to ori.csv
                    with open(dest_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['id', 'NMR'])  # Header
                        writer.writerows(valid_nmr_data)  # Data rows

                    logger.info(f"✅ Successfully saved {len(valid_nmr_data)} valid NMR entries to: {dest_path}")

                    # Verify file was created
                    if dest_path.exists():
                        logger.info(f"✅ File verification passed: {dest_path} exists")
                    else:
                        logger.error(f"❌ File verification failed: {dest_path} does not exist after writing")

                except Exception as e:
                    logger.error(f"Error saving processed CSV file: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")

            # NOW submit to Celery after CSV files are created
            try:
                celery_result = process_nmr_task.delay(str(task.task_id))
                logger.info(f"Task {task.task_id} submitted to Celery with result ID: {celery_result.id}")
            except Exception as celery_error:
                logger.error(f"Error submitting task to Celery: {str(celery_error)}")
                # 即使Celery提交失败，也继续处理，因为任务已经保存

            # 发送任务提交确认邮件
            if email:
                try:
                    send_task_confirmation_email(task)
                    logger.info(f"任务确认邮件已发送至 {email}")
                except Exception as e:
                    logger.error(f"发送任务确认邮件失败: {str(e)}")

            # Store task info in session for queue page notification
            request.session['task_id'] = str(task.task_id)
            request.session['task_name'] = task.task_name
            request.session.save()

            # 统一返回 JSON（登录体系下按账号隔离任务，不再生成/下载 token 文件）
            return JsonResponse({
                'success': True,
                'task_id': str(task.task_id),
                'task_name': task.task_name,
                'total_count': total_count,
                'message': 'Task submitted successfully',
            })

        except Exception as e:
            # Check if this was a file upload request by checking if input_file variable exists
            is_file_upload = 'input_file' in locals() and input_file

            if is_file_upload:
                # Return JSON error response for AJAX file upload requests
                return JsonResponse({
                    'error': f'Error creating task: {str(e)}'
                }, status=500)
            else:
                # Return HTML error page for form submissions
                return render(request, 'core_detection/submit.html', {
                    'error': f'Error creating task: {str(e)}',
                    'page_title': 'Submit New Task'
                })
    
    context = {
        'page_title': 'Submit New Task',
        'task_types': NMRTask.TASK_TYPE_CHOICES
    }
    return render(request, 'core_detection/submit.html', context)


@login_required
def task_queue(request):
    """Task queue page"""
    # Get filter parameters
    status_filter = request.GET.get('status', 'all')
    search_query = request.GET.get('search', '')

    # Base queryset - 只显示当前用户自己的任务（数据隔离）
    tasks_queryset = NMRTask.objects.select_related('submitted_by').filter(submitted_by=request.user)
    
    # Apply filters
    if status_filter != 'all':
        tasks_queryset = tasks_queryset.filter(status=status_filter)
    
    if search_query:
        tasks_queryset = tasks_queryset.filter(
            Q(task_name__icontains=search_query) |
            Q(task_id__icontains=search_query)
        )
    
    # Order by submission time
    tasks_queryset = tasks_queryset.order_by('-submitted_at')
    
    # Pagination
    paginator = Paginator(tasks_queryset, 20)  # 20 tasks per page
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
    
    # Statistics - 仅统计当前用户的任务
    stats = {
        'all': tasks_queryset.model.objects.filter(submitted_by=request.user).count(),
        'pending': tasks_queryset.model.objects.filter(submitted_by=request.user, status='pending').count(),
        'processing': tasks_queryset.model.objects.filter(submitted_by=request.user, status='processing').count(),
        'completed': tasks_queryset.model.objects.filter(submitted_by=request.user, status='completed').count(),
        'failed': tasks_queryset.model.objects.filter(submitted_by=request.user, status='failed').count(),
    }
    
    context = {
        'page_title': 'Task Queue',
        'page_obj': page_obj,
        'status_filter': status_filter,
        'search_query': search_query,
        'stats': stats,
        'status_choices': NMRTask.TASK_STATUS_CHOICES
    }
    return render(request, 'core_detection/queue.html', context)


@login_required
def task_detail(request, task_id):
    """Task detail page with token verification (disabled for testing)"""
    task = get_user_task_or_404(request, task_id)
    
    # TODO: Enable token verification for production
    # Check if user has verified access to this task
    # verified_tasks = request.session.get('verified_tasks', [])
    # if str(task_id) not in verified_tasks:
    #     # User needs to verify with token
    #     if request.method == 'POST':
    #         token_file = request.FILES.get('token_file')
    #         if not token_file:
    #             return render(request, 'core_detection/token_verification.html', {
    #                 'task': task,
    #                 'error': 'Token file is required'
    #             })
    #         
    #         # Read and verify token
    #         token_content = read_token_from_file(token_file)
    #         
    #         if verify_task_token(token_content, task.task_id):
    #             # Add to verified tasks
    #             verified_tasks.append(str(task_id))
    #             request.session['verified_tasks'] = verified_tasks
    #             # Continue to show task detail
    #         else:
    #             return render(request, 'core_detection/token_verification.html', {
    #                 'task': task,
    #                 'error': 'Invalid token. Please upload the correct token file for this task.'
    #             })
    #     else:
    #         # Show token verification page
    #         return render(request, 'core_detection/token_verification.html', {
    #             'task': task
    #         })
    
    # User has verified access, show task detail
    logs = task.logs.all().order_by('-timestamp')
    
    # Calculate NMR statistics if result data exists
    nmr_stats = {}
    if task.result_data and 'nmr_shifts' in task.result_data:
        nmr_shifts = task.result_data['nmr_shifts']
        nmr_stats = {
            'min': min(nmr_shifts) if nmr_shifts else 0,
            'max': max(nmr_shifts) if nmr_shifts else 0,
            'count': len(nmr_shifts),
            'sum': sum(nmr_shifts) if nmr_shifts else 0,
            'mean': (sum(nmr_shifts) / len(nmr_shifts)) if nmr_shifts else 0
        }
    
    context = {
        'page_title': f'Task Details: {task.task_name}',
        'task': task,
        'logs': logs,
        'nmr_stats': nmr_stats
    }
    return render(request, 'core_detection/detail.html', context)


@login_required
def task_result(request, task_id):
    """Task result page"""
    task = get_user_task_or_404(request, task_id)
    
    # Get the id parameter from URL (default to 1 if not provided)
    try:
        data_id = int(float(request.GET.get('id', 1)))  # Handle both int and float strings
    except (ValueError, TypeError):
        data_id = 1
    
    # Initialize variables for specific data
    specific_nmr_shift = None
    specific_predictions = None
    prediction_result = None
    
    if task.status == 'completed' and task.result_data:
        # Get the specific NMR shift based on id (id-1 because array is 0-indexed)
        specific_nmr_shift = None
        
        # Handle batch mode NMR shifts
        if task.result_data.get('batch_mode') and 'sample_results' in task.result_data:
            sample_results = task.result_data.get('sample_results', [])
            if 0 <= data_id - 1 < len(sample_results):
                sample_data = sample_results[data_id - 1]
                # Get NMR shifts for this specific sample (already a list)
                specific_nmr_shift = sample_data.get('nmr_shifts', [])
        else:
            # Handle legacy single sample mode
            if 'nmr_shifts' in task.result_data and task.result_data['nmr_shifts']:
                nmr_shifts = task.result_data['nmr_shifts']
                if 0 <= data_id - 1 < len(nmr_shifts):
                    specific_nmr_shift = nmr_shifts[data_id - 1]
        
        # Get prediction results if available
        all_predictions = []
        
        # Handle batch mode predictions
        if task.result_data.get('batch_mode') and 'sample_results' in task.result_data:
            sample_results = task.result_data.get('sample_results', [])
            if 0 <= data_id - 1 < len(sample_results):
                sample_data = sample_results[data_id - 1]
                # Get predictions from individual sample
                if 'cmgnet_smiles' in sample_data and 'predictions' in sample_data['cmgnet_smiles']:
                    all_predictions = sample_data['cmgnet_smiles']['predictions']
        else:
            # Handle legacy mode predictions
            if 'cmgnet_smiles' in task.result_data and 'predictions' in task.result_data['cmgnet_smiles']:
                all_predictions = task.result_data['cmgnet_smiles']['predictions']
        
        if all_predictions and len(all_predictions) > 0:
            # For now, use the first prediction or create a mock prediction based on the specific shift
            specific_predictions = all_predictions[0] if all_predictions else None
        
        # Create a prediction result object for the specific NMR shift
        if specific_nmr_shift is not None:
            # Handle batch mode results
            if task.result_data.get('batch_mode') and 'sample_results' in task.result_data:
                sample_results = task.result_data.get('sample_results', [])
                if 0 <= data_id - 1 < len(sample_results):
                    sample_data = sample_results[data_id - 1]
                    ensemble_result = sample_data.get('integrated_decision', {})
                    prediction_result = {
                        'id': data_id,
                        'nmr_shift': specific_nmr_shift,
                        'predictions': specific_predictions,
                        'ensemble_result': ensemble_result,
                        'xgboost_result': sample_data.get('xgboost', {}),
                        'cmgnet_frag_result': sample_data.get('cmgnet_frag', {}),
                        'cmgnet_smiles_result': sample_data.get('cmgnet_smiles', {}),
                        'total_shifts': len(sample_results),
                        'all_nmr_shifts': specific_nmr_shift if isinstance(specific_nmr_shift, list) else [specific_nmr_shift],  # Current sample NMR shifts for chart
                        'positive_votes': ensemble_result.get('positive_votes', 0),
                        'total_votes': ensemble_result.get('total_votes', 3)
                    }
                else:
                    prediction_result = None
            else:
                # Handle legacy single sample mode
                ensemble_result = task.result_data.get('integrated_decision', {})
                prediction_result = {
                    'id': data_id,
                    'nmr_shift': specific_nmr_shift,
                    'predictions': specific_predictions,
                    'ensemble_result': ensemble_result,
                    'xgboost_result': task.result_data.get('xgboost', {}),
                    'cmgnet_frag_result': task.result_data.get('cmgnet_frag', {}),
                    'cmgnet_smiles_result': task.result_data.get('cmgnet_smiles', {}),
                    'total_shifts': len(task.result_data.get('nmr_shifts', [])),
                    'all_nmr_shifts': task.result_data.get('nmr_shifts', []),
                    'positive_votes': ensemble_result.get('positive_votes', 0),
                    'total_votes': ensemble_result.get('total_votes', 3)
                }
    
    # Mock results data for demonstration (backward compatibility)
    results = []
    if task.status == 'completed' and task.result_data:
        if 'cores' in task.result_data:
            results = task.result_data['cores']
    
    context = {
        'page_title': f'Analysis Results: {task.task_name}',
        'task': task,
        'results': results,
        'data_id': data_id,
        'prediction_result': prediction_result,
        'specific_nmr_shift': specific_nmr_shift
    }
    return render(request, 'core_detection/result.html', context)


def contact(request):
    """Contact page"""
    if request.method == 'POST':
        try:
            name = request.POST.get('name')
            email = request.POST.get('email')
            message = request.POST.get('message')
            
            ContactMessage.objects.create(
                name=name,
                email=email,
                subject='',  # Empty string for removed subject field
                message=message
            )
            
            return render(request, 'core_detection/contact.html', {
                'success': True,
                'page_title': 'Contact Us'
            })
            
        except Exception as e:
            return render(request, 'core_detection/contact.html', {
                'error': f'Error sending message: {str(e)}',
                'page_title': 'Contact Us'
            })
    
    context = {
        'page_title': 'Contact Us'
    }
    return render(request, 'core_detection/contact.html', context)


# API Views
@login_required
def api_task_status(request, task_id):
    """API endpoint to get task status"""
    task = get_user_task_or_404(request, task_id)
    
    # Get Celery task status if available
    celery_status = None
    if task.celery_task_id:
        from celery.result import AsyncResult
        celery_task = AsyncResult(task.celery_task_id)
        celery_status = {
            'state': celery_task.state,
            'info': celery_task.info
        }
    
    data = {
        'task_id': task.task_id,
        'task_name': task.task_name,
        'status': task.status,
        'progress': task.progress_percentage,
        'current_step': task.current_step,
        'submitted_at': task.submitted_at.isoformat(),
        'processing_start_time': task.processing_start_time.isoformat() if task.processing_start_time else None,
        'processing_end_time': task.processing_end_time.isoformat() if task.processing_end_time else None,
        'processing_duration': task.processing_duration,
        'celery_status': celery_status,
        'result_data': task.result_data if task.status == 'completed' else None,
        'error_message': task.error_message if task.status == 'failed' else None
    }
    
    return JsonResponse(data)


@login_required
def api_queue_stats(request):
    """API endpoint to get queue statistics (当前用户自己的任务)"""
    user_tasks = NMRTask.objects.filter(submitted_by=request.user)
    stats = {
        'pending': user_tasks.filter(status='pending').count(),
        'processing': user_tasks.filter(status='processing').count(),
        'completed': user_tasks.filter(status='completed').count(),
        'failed': user_tasks.filter(status='failed').count(),
        'total': user_tasks.count(),
        'timestamp': timezone.now().isoformat()
    }
    
    return JsonResponse(stats)


@login_required
def api_recent_tasks(request):
    """API endpoint to get recent tasks (仅当前用户的)"""
    limit = int(request.GET.get('limit', 10))
    tasks = NMRTask.objects.select_related('submitted_by').filter(
        submitted_by=request.user
    ).order_by('-submitted_at')[:limit]
    
    data = []
    for task in tasks:
        profile = getattr(task.submitted_by, 'profile', None)
        data.append({
            'task_id': task.task_id,
            'task_name': task.task_name,
            'status': task.status,
            'progress': task.progress_percentage,
            'submitted_by': (profile.display_name if profile and profile.display_name
                             else task.submitted_by.username),
            'submitted_at': task.submitted_at.isoformat(),
            'task_type': task.task_type
        })
    
    return JsonResponse({'tasks': data})


def admin_redirect(request):
    """Redirect admin URLs to home page"""
    return redirect('core_detection:home')


@login_required
@login_required
def delete_task(request, task_id):
    """Delete task (仅任务所有者)"""
    if request.method == 'POST':
        try:
            task = get_user_task_or_404(request, task_id)
            
            # TODO: Enable token verification for production
            # token_file = request.FILES.get('token_file')
            # if not token_file:
            #     return JsonResponse({'success': False, 'error': 'Token file is required'})
            # 
            # # Read and verify token
            # token_content = read_token_from_file(token_file)
            # 
            # if verify_task_token(token_content, task.task_id):
            #     task.delete()
            #     return JsonResponse({'success': True, 'message': 'Task deleted successfully'})
            # else:
            #     return JsonResponse({'success': False, 'error': 'Invalid token'})
            
            # For testing: delete without token verification
            task.delete()
            return JsonResponse({'success': True, 'message': 'Task deleted successfully'})

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    else:
        return JsonResponse({'success': False, 'error': 'Invalid request method'})


@login_required
def download_task_result(request, task_id):
    """Download task result JSON file"""
    try:
        task = get_user_task_or_404(request, task_id)

        if task.status != 'completed':
            return JsonResponse({'error': 'Task is not completed yet'}, status=400)

        if not task.result_data:
            return JsonResponse({'error': 'No result data available'}, status=404)

        # Create JSON response with the result data
        import json
        from django.http import HttpResponse

        result_json = json.dumps(task.result_data, indent=2, ensure_ascii=False)
        response = HttpResponse(result_json, content_type='application/json')
        response['Content-Disposition'] = f'attachment; filename="task_{task_id}_results.json"'

        logger.info(f"Task result downloaded: {task_id}")
        return response

    except Exception as e:
        logger.error(f"Error downloading task result: {str(e)}")
        return JsonResponse({'error': f'Error downloading result: {str(e)}'}, status=500)


@login_required
def download_batch_results(request, task_id):
    """Download result.json file from the result directory"""
    try:
        task = get_user_task_or_404(request, task_id)

        if task.status != 'completed':
            return JsonResponse({'error': 'Task is not completed yet'}, status=400)

        if not task.celery_task_id:
            return JsonResponse({'error': 'No Celery task ID available'}, status=404)

        # Construct the path to the result.json file
        result_dir = f'/home/lingjiebao/project/NMR_Core_Mol_Detection/result/{task.celery_task_id}'
        result_file_path = os.path.join(result_dir, 'result.json')

        # Check if the result file exists
        if not os.path.exists(result_file_path):
            return JsonResponse({'error': f'Result file not found: {result_file_path}'}, status=404)

        # Create filename for download
        download_filename = f"result_{task.task_id}.json"

        # Create file response for download
        with open(result_file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/json')
            response['Content-Disposition'] = f'attachment; filename="{download_filename}"'
            response['Content-Length'] = os.path.getsize(result_file_path)

        logger.info(f"Result file downloaded: {task_id} - {download_filename} from {result_file_path}")
        return response

    except Exception as e:
        logger.error(f"Error downloading result file: {str(e)}")
        return JsonResponse({'error': f'Error downloading result file: {str(e)}'}, status=500)


@login_required
def download_single_result(request, task_id, result_id):
    """Download single result based on result ID as JSON file"""
    try:
        task = get_user_task_or_404(request, task_id)

        if task.status != 'completed':
            return JsonResponse({'error': 'Task is not completed yet'}, status=400)

        if not task.result_data:
            return JsonResponse({'error': 'No result data available'}, status=404)

        result_id_int = int(result_id)
        single_result = None

        # Check if this is batch mode or single molecule mode
        if task.result_data.get('batch_mode'):
            # Handle batch mode - get specific sample from sample_results
            sample_results = task.result_data.get('sample_results', [])
            
            # Find the sample with matching sample_id (which is result_id)
            target_sample = None
            for sample in sample_results:
                if sample.get('sample_id') == result_id_int:
                    target_sample = sample
                    break
            
            if not target_sample:
                return JsonResponse({'error': f'Sample ID {result_id} not found in batch results'}, status=404)

            # Create single result data structure for batch mode
            single_result = {
                'result_id': result_id_int,
                'task_id': str(task.task_id),
                'task_name': task.task_name,
                'batch_mode': True,
                'sample_id': target_sample.get('sample_id'),
                'nmr_count': target_sample.get('nmr_count'),
                'nmr_shifts': target_sample.get('nmr_shifts', []),
                'xgboost_result': target_sample.get('xgboost', {}),
                'cmgnet_frag_result': target_sample.get('cmgnet_frag', {}),
                'cmgnet_smiles_result': target_sample.get('cmgnet_smiles', {}),
                'integrated_decision': target_sample.get('integrated_decision', {}),
                'parameters': {
                    'xgboost_threshold': target_sample.get('xgboost_threshold'),
                    'topk': target_sample.get('topk'),
                    'prediction_mode': target_sample.get('prediction_mode')
                },
                'generated_at': task.processing_end_time.isoformat() if task.processing_end_time else None
            }
        else:
            # Handle single molecule mode
            if 'nmr_shifts' in task.result_data and task.result_data['nmr_shifts']:
                nmr_shifts = task.result_data['nmr_shifts']
                if 0 <= result_id_int - 1 < len(nmr_shifts):
                    specific_nmr_shift = nmr_shifts[result_id_int - 1]
                else:
                    return JsonResponse({'error': f'Result ID {result_id} not found'}, status=404)
            else:
                return JsonResponse({'error': 'No NMR shifts data available'}, status=404)

            # Create single result data structure for single molecule mode
            single_result = {
                'result_id': result_id_int,
                'task_id': str(task.task_id),
                'task_name': task.task_name,
                'batch_mode': False,
                'nmr_shift': specific_nmr_shift,
                'ensemble_result': task.result_data.get('integrated_decision', {}),
                'xgboost_result': task.result_data.get('xgboost', {}),
                'cmgnet_frag_result': task.result_data.get('cmgnet_frag', {}),
                'cmgnet_smiles_result': task.result_data.get('cmgnet_smiles', {}),
                'all_nmr_shifts': task.result_data.get('nmr_shifts', []),
                'total_shifts': len(task.result_data.get('nmr_shifts', [])),
                'generated_at': task.processing_end_time.isoformat() if task.processing_end_time else None
            }

        # Create JSON response as file download
        import json
        from django.http import HttpResponse

        result_json = json.dumps(single_result, indent=2, ensure_ascii=False)
        response = HttpResponse(result_json, content_type='application/json')
        response['Content-Disposition'] = f'attachment; filename="result_{task.task_id}_id{result_id_int}.json"'

        logger.info(f"Single result downloaded: {task.task_id}, result ID: {result_id_int}")
        return response

    except Exception as e:
        logger.error(f"Error downloading single result: {str(e)}")
        return JsonResponse({'error': f'Error downloading result: {str(e)}'}, status=500)