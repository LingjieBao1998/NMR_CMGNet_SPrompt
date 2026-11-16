import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'nmr_core_system.settings')

app = Celery('nmr_core_system')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()