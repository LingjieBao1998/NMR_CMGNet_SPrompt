from django.db import models
from django.contrib.auth.models import User
import hashlib
from datetime import datetime


class NMRTask(models.Model):
    """NMR Core Detection Task"""
    TASK_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    TASK_TYPE_CHOICES = [
        ('single_molecule', 'Single Molecule Detection'),
        ('batch_detection', 'Batch Core Detection'),
        ('custom_analysis', 'Custom Analysis'),
    ]
    
    # Basic Information
    task_id = models.CharField(max_length=100, unique=True, verbose_name="Task ID")
    task_name = models.CharField(max_length=255, verbose_name="Task Name")
    task_type = models.CharField(
        max_length=20, 
        choices=TASK_TYPE_CHOICES,
        default='single_molecule',
        verbose_name="Task Type"
    )
    
    # User Information
    submitted_by = models.ForeignKey(
        User, 
        on_delete=models.CASCADE,
        verbose_name="Submitted By"
    )
    submitted_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Submitted At"
    )
    email = models.EmailField(
        null=True,
        blank=True,
        verbose_name="Email"
    )
    user_name = models.CharField(
        max_length=255,
        default='',
        blank=True,
        verbose_name="用户名"
    )
    task_token = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        verbose_name="Task Token"
    )
    
    # Task Status
    status = models.CharField(
        max_length=20,
        choices=TASK_STATUS_CHOICES,
        default='pending',
        verbose_name="Status"
    )
    
    # Celery Task Information
    celery_task_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        verbose_name="Celery Task ID"
    )
    
    # Processing Information
    processing_start_time = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name="Processing Start Time"
    )
    processing_end_time = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name="Processing End Time"
    )
    processing_duration = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Processing Duration (seconds)"
    )
    
    # Input Data
    input_nmr = models.TextField(
        blank=True,
        verbose_name="Input NMR"
    )
    input_file = models.FileField(
        upload_to='uploads/nmr_input/%Y/%m/%d/',
        null=True,
        blank=True,
        verbose_name="Input File"
    )
    parameters = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="Task Parameters"
    )
    
    # Output Results
    result_data = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="Result Data"
    )
    output_file = models.FileField(
        upload_to='outputs/nmr_results/%Y/%m/%d/',
        null=True,
        blank=True,
        verbose_name="Output File"
    )
    
    # Error Information
    error_message = models.TextField(
        blank=True,
        verbose_name="Error Message"
    )
    error_details = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="Error Details"
    )
    
    # Progress Tracking
    progress_percentage = models.IntegerField(
        default=0,
        verbose_name="Progress Percentage"
    )
    current_step = models.CharField(
        max_length=255,
        blank=True,
        verbose_name="Current Step"
    )
    
    # Detection Results
    total_count = models.IntegerField(
        default=0,
        verbose_name="Total Count"
    )
    positive_count = models.IntegerField(
        default=0,
        verbose_name="Positive Count"
    )
    
    class Meta:
        verbose_name = "NMR Task"
        verbose_name_plural = "NMR Tasks"
        ordering = ['-submitted_at']
    
    def __str__(self):
        return f"{self.task_name} - {self.get_status_display()}"
    
    def save(self, *args, **kwargs):
        # Generate task ID if not exists
        if not self.task_id:
            import uuid
            from django.db import IntegrityError
            max_retries = 5
            retry_count = 0

            while retry_count < max_retries:
                # 使用UUID和时间戳确保唯一性
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                self.task_id = f"NMR_{timestamp}_{unique_id}"

                try:
                    # Try to save and break if successful
                    super().save(*args, **kwargs)
                    return
                except IntegrityError:
                    # If task_id already exists, try again with a new ID
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise IntegrityError(f"Failed to generate unique task ID after {max_retries} attempts")
                    continue

        # Calculate processing duration if both times are available
        if self.processing_start_time and self.processing_end_time:
            delta = self.processing_end_time - self.processing_start_time
            self.processing_duration = delta.total_seconds()

        super().save(*args, **kwargs)


class TaskLog(models.Model):
    """Task Processing Log"""
    LOG_LEVEL_CHOICES = [
        ('info', 'Info'),
        ('warning', 'Warning'),
        ('error', 'Error'),
        ('debug', 'Debug'),
    ]
    
    task = models.ForeignKey(
        NMRTask,
        on_delete=models.CASCADE,
        related_name='logs',
        verbose_name="Task"
    )
    log_level = models.CharField(
        max_length=10,
        choices=LOG_LEVEL_CHOICES,
        default='info',
        verbose_name="Log Level"
    )
    message = models.TextField(verbose_name="Message")
    timestamp = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Timestamp"
    )
    details = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="Details"
    )
    
    class Meta:
        verbose_name = "Task Log"
        verbose_name_plural = "Task Logs"
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.task.task_name} - {self.get_log_level_display()}: {self.message[:50]}"


class ContactMessage(models.Model):
    """Contact Form Messages"""
    name = models.CharField(max_length=255, verbose_name="Name")
    email = models.EmailField(verbose_name="Email")
    subject = models.CharField(max_length=255, verbose_name="Subject")
    message = models.TextField(verbose_name="Message")
    submitted_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Submitted At"
    )
    is_read = models.BooleanField(
        default=False,
        verbose_name="Is Read"
    )
    
    class Meta:
        verbose_name = "Contact Message"
        verbose_name_plural = "Contact Messages"
        ordering = ['-submitted_at']
    
    def __str__(self):
        return f"{self.name} - {self.subject}"


class SystemStatistics(models.Model):
    """System Usage Statistics"""
    date = models.DateField(
        unique=True,
        verbose_name="Date"
    )
    total_tasks = models.IntegerField(
        default=0,
        verbose_name="Total Tasks"
    )
    completed_tasks = models.IntegerField(
        default=0,
        verbose_name="Completed Tasks"
    )
    failed_tasks = models.IntegerField(
        default=0,
        verbose_name="Failed Tasks"
    )
    active_users = models.IntegerField(
        default=0,
        verbose_name="Active Users"
    )
    avg_processing_time = models.FloatField(
        default=0.0,
        verbose_name="Average Processing Time (seconds)"
    )
    
    class Meta:
        verbose_name = "System Statistics"
        verbose_name_plural = "System Statistics"
        ordering = ['-date']
    
    def __str__(self):
        return f"Statistics for {self.date}"


class PageVisit(models.Model):
    """页面访问统计"""
    page_name = models.CharField(max_length=100, help_text="页面名称")
    visit_count = models.IntegerField(default=0, help_text="访问次数")
    last_visit = models.DateTimeField(auto_now=True, help_text="最后访问时间")

    class Meta:
        verbose_name = "页面访问统计"
        verbose_name_plural = "页面访问统计"

    def __str__(self):
        return f"{self.page_name}: {self.visit_count} 次访问"


class UserProfile(models.Model):
    """用户资料：邮箱验证与审批状态

    注册流程：注册(is_active=False, email_verified=False, is_approved=False)
    -> 邮箱验证(email_verified=True) -> 管理员审批(is_approved=True 且 is_active=True)
    -> 可登录使用。
    is_active 由 Django auth 控制：验证邮件前为 False，管理员批准时置 True，
    这样未批准用户即使知道密码也无法通过 authenticate()。
    """
    APPROVAL_STATUS_CHOICES = [
        ('pending', 'Pending Approval'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='profile',
        verbose_name="User"
    )
    display_name = models.CharField(
        max_length=255,
        default='',
        blank=True,
        verbose_name="显示名（可重复）"
    )
    email_verified = models.BooleanField(
        default=False,
        verbose_name="Email Verified"
    )
    email_verification_token = models.CharField(
        max_length=64,
        null=True,
        blank=True,
        verbose_name="Email Verification Token"
    )
    password_reset_token = models.CharField(
        max_length=64,
        null=True,
        blank=True,
        verbose_name="Password Reset Token"
    )
    password_reset_token_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name="Password Reset Token Created At"
    )
    approval_status = models.CharField(
        max_length=10,
        choices=APPROVAL_STATUS_CHOICES,
        default='pending',
        verbose_name="Approval Status"
    )
    institution = models.CharField(
        max_length=255,
        default='',
        blank=True,
        verbose_name="机构/单位"
    )
    registered_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Registered At"
    )
    approved_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name="Approved At"
    )

    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"

    def __str__(self):
        return f"{self.user.username} ({self.approval_status})"

    @property
    def is_usable(self):
        """邮箱已验证且已审批"""
        return self.email_verified and self.approval_status == 'approved'