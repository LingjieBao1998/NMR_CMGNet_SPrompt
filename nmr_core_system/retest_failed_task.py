#!/usr/bin/env python
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core_detection.settings')
django.setup()
from core_detection.models import NMRTask

task_id = 'NMR_20260615110113_f6bde850'
print(f"Resetting and retesting task: {task_id}")

try:
    task = NMRTask.objects.get(task_id=task_id)
    print(f"Current status: {task.status}")
    
    # Reset task to pending status
    task.status = 'pending'
    task.error_message = ""
    task.error_details = {}
    task.processing_start_time = None
    task.processing_end_time = None
    task.processing_duration = None
    task.positive_count = 0
    task.total_count = 0
    task.progress_percentage = 0
    if task.result_data is None:
        task.result_data = {}
    task.save()
    
    print(f"✅ Task reset to pending status")
    print(f"Task input_nmr preview: {task.input_nmr[:100] if task.input_nmr else 'None'}...")
    
    # Trigger task processing through Celery
    from core_detection.tasks import process_nmr_task
    print(f"🔄 Triggering task processing through Celery...")
    
    # Use celery_task.delay() instead of direct call
    result = process_nmr_task.delay(task_id)
    print(f"✅ Task scheduled for processing")
    print(f"Celery task ID: {result.id}")
    
    # Wait a bit and check status
    import time
    time.sleep(5)
    
    task.refresh_from_db()
    print(f"Task status after 5 seconds: {task.status}")
    if task.error_message:
        print(f"Error: {task.error_message}")
        
except NMRTask.DoesNotExist:
    print(f"❌ Task not found: {task_id}")
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()

