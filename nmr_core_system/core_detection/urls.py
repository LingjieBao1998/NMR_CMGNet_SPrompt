from django.urls import path
from . import views
from . import auth_views

app_name = 'core_detection'

urlpatterns = [
    # Authentication
    path('register/', auth_views.register_view, name='register'),
    path('login/', auth_views.login_view, name='login'),
    path('logout/', auth_views.logout_view, name='logout'),
    path('verify-email/<str:token>/', auth_views.verify_email_view, name='verify_email'),
    path('forgot-password/', auth_views.forgot_password_view, name='forgot_password'),
    path('reset-password/<str:token>/', auth_views.reset_password_view, name='reset_password'),
    path('api/check-email/', auth_views.api_check_email, name='api_check_email'),

    # Main pages
    path('', views.home, name='home'),
    path('submission/', views.submit_task, name='submit_task'),
    path('queue/', views.task_queue, name='task_queue'),
    path('task/<str:task_id>/', views.task_detail, name='task_detail'),
    path('task/<str:task_id>/result/', views.task_result, name='task_result'),
    path('contact/', views.contact, name='contact'),
    
    # Task management
    path('task/<str:task_id>/download/', views.download_task_result, name='download_task_result'),
    path('task/<str:task_id>/download-batch-results/', views.download_batch_results, name='download_batch_results'),
    path('task/<str:task_id>/result/<int:result_id>/download/', views.download_single_result, name='download_single_result'),
    path('task/<str:task_id>/delete/', views.delete_task, name='delete_task'),

    # API endpoints
    path('api/task-status/<str:task_id>/', views.api_task_status, name='api_task_status'),
    path('api/queue-stats/', views.api_queue_stats, name='api_queue_stats'),
    path('api/recent-tasks/', views.api_recent_tasks, name='api_recent_tasks'),
]