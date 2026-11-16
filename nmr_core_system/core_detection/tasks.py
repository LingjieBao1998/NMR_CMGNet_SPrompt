import time
import logging
from celery import shared_task
from django.utils import timezone
from django.core.files.base import ContentFile
from .models import NMRTask, TaskLog
from .email_utils import send_task_result_email, send_task_failed_email
import json
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

@shared_task(bind=True, name='core_detection.tasks.process_nmr_task')
def process_nmr_task(self, task_id):
    """
    Asynchronous task for processing NMR core detection
    """
    logger.info(f"Starting NMR task processing: {task_id}")
    
    try:
        # Get the task from database
        try:
            task = NMRTask.objects.get(task_id=task_id)
            task.status = 'processing'
            task.processing_start_time = timezone.now()

            # Update celery_task_id if not set
            if not task.celery_task_id:
                task.celery_task_id = self.request.id
                task.save()

            logger.info(f"Found task: {task.task_name}, celery_task_id: {task.celery_task_id}")
        except NMRTask.DoesNotExist:
            logger.error(f"Task not found: {task_id}")
            raise Exception(f"Task {task_id} not found")

        # Get task parameters
        parameters = task.parameters
        model_selection = parameters.get('model_selection', 'ensemble')
        xgboost_threshold = parameters.get('xgboost_threshold', 0.5)
        topk = parameters.get('topk', 10)

        # Create initial log
        TaskLog.objects.create(
            task=task,
            log_level='info',
            message='Task processing started',
            details={'step': 'start', 'parameters': parameters}
        )
        
        # Update task status
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Initializing...'}
        )
        
        # Import the ensemble predictor
        from .ensemble_nmr_predictor import EnsembleNMRPredictor

        # Initialize predictor
        predictor = EnsembleNMRPredictor()

        # Parse NMR data - support both direct input and CSV file
        nmr_data = []
        
        try:
            # First, try to use direct NMR input if available
            if task.input_nmr and task.input_nmr.strip():
                logger.info("Using direct NMR input from task")
                nmr_text = task.input_nmr.strip()
                
                # Parse comma-separated values
                values = []
                if ',' in nmr_text:
                    values = [v.strip() for v in nmr_text.split(',') if v.strip()]
                else:
                    values = [v.strip() for v in nmr_text.split() if v.strip()]
                
                # Convert to float values
                float_values = []
                for val in values:
                    try:
                        float_val = float(val)
                        # Validate NMR chemical shift range (-10 to 350 ppm)
                        if -10 <= float_val <= 350:
                            float_values.append(float_val)
                        else:
                            logger.warning(f"NMR value {float_val} out of valid range (-10 to 350 ppm), skipping")
                    except ValueError:
                        logger.warning(f"Could not convert value '{val}' to float, skipping")
                        continue
                
                if float_values:
                    nmr_data.append(float_values)
                    logger.info(f"Successfully parsed {len(float_values)} NMR values from direct input")
                else:
                    raise ValueError("No valid NMR data found in direct input. Please provide 13C NMR chemical shifts (-10 to 350 ppm).")
            
            # If no direct input or parsing failed, try CSV file
            else:
                logger.info("No direct NMR input, trying CSV file processing")
                
                # Determine CSV file path - consistent with views.py
                project_dir = Path(__file__).parent.parent  # Points to /app/nmr_core_system
                csv_file_path = project_dir / "result" / task.celery_task_id / "ori.csv"
                
                if not csv_file_path.exists():
                    error_msg = f"No NMR input provided and CSV file not found: {csv_file_path}. Please provide either direct NMR data or upload a CSV file."
                    logger.error(error_msg)
                    # Mark task as failed and return early
                    task.status = 'failed'
                    task.processing_end_time = timezone.now()
                    task.error_message = error_msg
                    task.save()
                    return {'status': 'failed', 'error': error_msg}
                
                logger.info(f"Reading NMR data from CSV file: {csv_file_path}")
                
                # Read CSV file
                import pandas as pd
                df = pd.read_csv(csv_file_path)
                
                # Find NMR column
                nmr_column = None
                for col in df.columns:
                    if 'nmr' in col.lower() or 'shift' in col.lower():
                        nmr_column = col
                        break
                
                if nmr_column is None:
                    # If no NMR column found, try second column
                    if len(df.columns) >= 2:
                        nmr_column = df.columns[1]
                    else:
                        raise ValueError("CSV file must have at least 2 columns (id, NMR)")
                
                # Parse NMR data from CSV column
                for idx, row in df.iterrows():
                    nmr_text = str(row[nmr_column]).strip()
                    
                    if not nmr_text:
                        continue
                    
                    # Parse comma-separated values
                    values = []
                    if ',' in nmr_text:
                        values = [v.strip() for v in nmr_text.split(',') if v.strip()]
                    else:
                        values = [v.strip() for v in nmr_text.split() if v.strip()]
                    
                    # Convert to float values
                    float_values = []
                    for val in values:
                        try:
                            float_val = float(val)
                            # Validate NMR chemical shift range (-10 to 350 ppm)
                            if -10 <= float_val <= 350:
                                float_values.append(float_val)
                            else:
                                logger.warning(f"NMR value {float_val} out of valid range (-10 to 350 ppm), skipping")
                        except ValueError:
                            logger.warning(f"Could not convert value '{val}' to float, skipping")
                            continue
                    
                    if float_values:
                        nmr_data.append(float_values)
                
                if not nmr_data:
                    raise ValueError("No valid NMR data found in CSV file")
                
                logger.info(f"Parsed {len(nmr_data)} samples from CSV file")

        except Exception as e:
            logger.error(f"Error processing NMR data: {str(e)}")
            # Mark task as failed and return early to avoid Celery serialization issues
            task = NMRTask.objects.get(task_id=task_id)
            task.status = 'failed'
            task.processing_end_time = timezone.now()
            task.error_message = str(e)
            task.save()
            return {'status': 'failed', 'error': str(e)}

        # Validate and format nmr_data
        if not nmr_data:
            raise ValueError("No valid NMR data found. Please provide 13C NMR chemical shifts (-10 to 350 ppm).")

        # nmr_data已经是从CSV解析好的list[list[float]]格式
        logger.info(f"Final NMR data format: list of {len(nmr_data)} sample(s)")
        logger.info(f"Sample 1: {nmr_data[0][:5]}... (first 5 values)")

        # Processing steps
        processing_steps = [
            (10, "Validating input data...", "validation"),
            (30, "Loading models...", "loading"),
            (50, "Running NMR prediction...", "prediction"),
            (80, "Analyzing results...", "analysis"),
            (95, "Finalizing analysis...", "finalization"),
            (100, "Processing completed!", "completed")
        ]

        result_data = {}

        for progress, message, step_name in processing_steps:
            logger.info(f"Processing step: {step_name} - {progress}% - {message}")
            
            # Update task progress
            task.progress_percentage = progress
            task.current_step = message
            task.save()
            
            # Create log for this step
            TaskLog.objects.create(
                task=task,
                log_level='info',
                message=message,
                details={'step': step_name, 'progress': progress}
            )
            
            # Update Celery task state
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': progress,
                    'total': 100,
                    'status': message,
                    'task_id': task_id,
                    'task_name': task.task_name
                }
            )

            # Execute actual NMR prediction at the prediction step
            if step_name == "prediction":
                logger.info(f"Running NMR prediction with {model_selection} model")
                try:
                    prediction_mode = "ensemble" if model_selection == "ensemble" else "xgboost_only"

                    # Call the ensemble predictor
                    prediction_result = predictor.predict(
                        nmr_list=nmr_data,
                        save_path=None,
                        celery_task_id=task.celery_task_id,
                        xgboost_threshold=float(xgboost_threshold),
                        num_returns=int(topk),
                        prediction_mode=prediction_mode
                    )

                    result_data = prediction_result
                    logger.info(f"Prediction completed successfully")

                except Exception as e:
                    logger.error(f"Prediction failed: {str(e)}")
                    # Mark task as failed and return early
                    task.status = 'failed'
                    task.processing_end_time = timezone.now()
                    task.error_message = f"NMR prediction failed: {str(e)}"
                    task.save()
                    return {'status': 'failed', 'error': f"NMR prediction failed: {str(e)}"}
            elif step_name == "analysis":
                # Add analysis metadata
                result_data['task_parameters'] = {
                    'model_selection': model_selection,
                    'xgboost_threshold': xgboost_threshold,
                    'topk': topk,
                    'num_samples': task.total_count  # Use the pre-calculated total_count
                }
            elif progress < 100:
                # Simulate processing time for other steps
                time.sleep(1)
        
        # Complete the task
        task.status = 'completed'
        task.processing_end_time = timezone.now()
        task.progress_percentage = 100
        task.current_step = 'Processing completed'
        task.result_data = result_data

        # Update detection counts based on batch results
        if result_data.get('batch_mode'):
            # 批量处理模式
            task.total_count = result_data.get('total_samples', len(nmr_data))
            task.positive_count = result_data.get('positive_samples', 0)
            logger.info(f"📊 Batch processing: {task.positive_count}/{task.total_count} samples detected as core molecules")
        else:
            # 单样本处理模式（向后兼容）
            task.total_count = len(nmr_data)
            if result_data.get('integrated_decision', {}).get('is_core_molecule'):
                task.positive_count = 1
            else:
                task.positive_count = 0
            logger.info(f"📊 Single sample processing: {task.positive_count}/{task.total_count} samples detected as core molecules")

        # Calculate processing duration
        if task.processing_start_time and task.processing_end_time:
            delta = task.processing_end_time - task.processing_start_time
            task.processing_duration = delta.total_seconds()
        
        task.save()
        
        # Create completion log
        TaskLog.objects.create(
            task=task,
            log_level='info',
            message=f'Task completed successfully in {task.processing_duration:.2f} seconds',
            details={
                'step': 'completion',
                'duration': task.processing_duration,
                'result_summary': result_data
            }
        )
        
        logger.info(f"NMR task processing completed: {task_id}")
        
        # 发送任务完成邮件
        if task.email:
            try:
                send_task_result_email(task)
                logger.info(f"任务结果邮件已发送至 {task.email}")
            except Exception as e:
                logger.error(f"发送任务结果邮件失败: {str(e)}")

        # Cleanup predictor resources
        try:
            if predictor:
                predictor.cleanup()
                logger.info("Predictor resources cleaned up successfully")
        except Exception as cleanup_error:
            logger.warning(f"Error during predictor cleanup: {str(cleanup_error)}")

        # Force garbage collection to free memory
        try:
            import gc
            gc.collect()
            logger.info("Garbage collection completed")
        except Exception as gc_error:
            logger.warning(f"Error during garbage collection: {str(gc_error)}")

        # Return result
        return {
            'status': 'success',
            'task_id': task_id,
            'task_name': task.task_name,
            'processing_duration': task.processing_duration,
            'result_data': result_data,
            'detected_cores': len(result_data.get('detected_cores', [])),
            'matched_patterns': len(result_data.get('matched_patterns', []))
        }
        
    except Exception as e:
        logger.error(f"Error processing NMR task: {str(e)}")

        # Cleanup predictor resources even on failure
        try:
            if 'predictor' in locals() and predictor:
                predictor.cleanup()
                logger.info("Predictor resources cleaned up after error")
        except Exception as cleanup_error:
            logger.warning(f"Error during predictor cleanup after failure: {str(cleanup_error)}")
            # Fallback to basic cleanup if cleanup method fails
            try:
                if 'predictor' in locals() and predictor:
                    del predictor
                    logger.info("Basic predictor cleanup completed")
            except:
                pass

        # Force garbage collection to free memory
        try:
            import gc
            gc.collect()
            logger.info("Garbage collection completed after error")
        except Exception as gc_error:
            logger.warning(f"Error during garbage collection after failure: {str(gc_error)}")

        # Update task status to failed
        try:
            task = NMRTask.objects.get(task_id=task_id)
            task.status = 'failed'
            task.processing_end_time = timezone.now()
            task.error_message = str(e)
            task.error_details = {
                'exception_type': type(e).__name__,
                'error_message': str(e)
            }
            
            # Calculate duration even for failed tasks
            if task.processing_start_time and task.processing_end_time:
                delta = task.processing_end_time - task.processing_start_time
                task.processing_duration = delta.total_seconds()
            
            task.save()
            
            # Create error log
            TaskLog.objects.create(
                task=task,
                log_level='error',
                message=f'Task failed: {str(e)}',
                details={
                    'step': 'error',
                    'exception_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            
            # 发送任务失败邮件
            if task.email:
                try:
                    send_task_failed_email(task)
                    logger.info(f"任务失败通知邮件已发送至 {task.email}")
                except Exception as email_error:
                    logger.error(f"发送任务失败通知邮件失败: {str(email_error)}")
        except:
            pass
        
        # Update Celery task state
        self.update_state(
            state='FAILURE',
            meta={
                'task_id': task_id,
                'error': str(e),
                'status': 'Processing failed'
            }
        )
        
        raise


@shared_task(name='core_detection.tasks.cleanup_old_tasks')
def cleanup_old_tasks(days=30):
    """
    Clean up old completed tasks
    """
    from datetime import timedelta
    from django.utils import timezone
    
    cutoff_date = timezone.now() - timedelta(days=days)
    old_tasks = NMRTask.objects.filter(
        status='completed',
        processing_end_time__lt=cutoff_date
    )
    
    count = old_tasks.count()
    old_tasks.delete()
    
    logger.info(f"Cleaned up {count} old tasks")
    return {'cleaned_tasks': count}


@shared_task(name='core_detection.tasks.update_system_statistics')
def update_system_statistics():
    """
    Update daily system statistics
    """
    from datetime import date
    from django.db.models import Avg, Count, Q
    
    today = date.today()
    
    # Get or create today's statistics
    stats, created = SystemStatistics.objects.get_or_create(
        date=today,
        defaults={
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'active_users': 0,
            'avg_processing_time': 0.0
        }
    )
    
    # Update statistics
    today_tasks = NMRTask.objects.filter(
        submitted_at__date=today
    )
    
    stats.total_tasks = today_tasks.count()
    stats.completed_tasks = today_tasks.filter(status='completed').count()
    stats.failed_tasks = today_tasks.filter(status='failed').count()
    stats.active_users = today_tasks.values('submitted_by').distinct().count()
    
    # Calculate average processing time for completed tasks
    completed_tasks = today_tasks.filter(
        status='completed',
        processing_duration__isnull=False
    )
    if completed_tasks.exists():
        stats.avg_processing_time = completed_tasks.aggregate(
            avg_time=Avg('processing_duration')
        )['avg_time'] or 0.0
    
    stats.save()
    
    logger.info(f"Updated system statistics for {today}")
    return {
        'date': str(today),
        'total_tasks': stats.total_tasks,
        'completed_tasks': stats.completed_tasks,
        'failed_tasks': stats.failed_tasks,
        'active_users': stats.active_users,
        'avg_processing_time': stats.avg_processing_time
    }