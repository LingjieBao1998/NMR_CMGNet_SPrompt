from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from django.utils import timezone

from .models import NMRTask, TaskLog, ContactMessage, SystemStatistics, PageVisit, UserProfile


class UserProfileInline(admin.StackedInline):
    """用户资料内联：邮箱验证 + 审批状态"""
    model = UserProfile
    can_delete = False
    fields = ('display_name', 'email_verified', 'approval_status', 'institution', 'registered_at', 'approved_at')
    readonly_fields = ('registered_at', 'approved_at')
    verbose_name_plural = 'User Profile（邮箱验证 / 审批）'


@admin.action(description='批准选中用户（可登录使用系统）')
def approve_users(modeladmin, request, queryset):
    for user in queryset:
        profile, _ = UserProfile.objects.get_or_create(user=user)
        profile.approval_status = 'approved'
        profile.approved_at = timezone.now()
        profile.save(update_fields=['approval_status', 'approved_at'])
        user.is_active = True
        user.save(update_fields=['is_active'])
    modeladmin.message_user(request, f"已批准 {queryset.count()} 个用户")


@admin.action(description='拒绝/禁用选中用户')
def reject_users(modeladmin, request, queryset):
    for user in queryset:
        profile, _ = UserProfile.objects.get_or_create(user=user)
        profile.approval_status = 'rejected'
        profile.save(update_fields=['approval_status'])
        user.is_active = False
        user.save(update_fields=['is_active'])
    modeladmin.message_user(request, f"已拒绝 {queryset.count()} 个用户")


class CustomUserAdmin(UserAdmin):
    """在用户管理页显示审批状态，支持批量批准"""
    inlines = (UserProfileInline,)
    list_display = ('username', 'email', 'is_active', 'get_approval_status', 'get_email_verified', 'date_joined')
    list_filter = ('is_active', 'date_joined')
    actions = [approve_users, reject_users]

    @admin.display(description='Approval')
    def get_approval_status(self, obj):
        profile = getattr(obj, 'profile', None)
        return profile.approval_status if profile else '-'

    @admin.display(description='Email Verified', boolean=True)
    def get_email_verified(self, obj):
        profile = getattr(obj, 'profile', None)
        return profile.email_verified if profile else False


# 用自定义 UserAdmin 替换默认注册
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)


@admin.register(NMRTask)
class NMRTaskAdmin(admin.ModelAdmin):
    list_display = ['task_id', 'task_name', 'task_type', 'status', 'submitted_by', 'progress_percentage', 'submitted_at']
    list_filter = ['status', 'task_type', 'submitted_at']
    search_fields = ['task_id', 'task_name', 'submitted_by__username']
    readonly_fields = ['task_id', 'submitted_at', 'processing_start_time', 'processing_end_time', 'processing_duration']
    date_hierarchy = 'submitted_at'


@admin.register(TaskLog)
class TaskLogAdmin(admin.ModelAdmin):
    list_display = ['task', 'log_level', 'message', 'timestamp']
    list_filter = ['log_level', 'timestamp']
    search_fields = ['task__task_id', 'task__task_name', 'message']
    readonly_fields = ['timestamp']


@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'subject', 'is_read', 'submitted_at']
    list_filter = ['is_read', 'submitted_at']
    search_fields = ['name', 'email', 'subject', 'message']
    readonly_fields = ['submitted_at']


@admin.register(SystemStatistics)
class SystemStatisticsAdmin(admin.ModelAdmin):
    list_display = ['date', 'total_tasks', 'completed_tasks', 'failed_tasks', 'active_users', 'avg_processing_time']
    list_filter = ['date']
    readonly_fields = ['date']
    date_hierarchy = 'date'


@admin.register(PageVisit)
class PageVisitAdmin(admin.ModelAdmin):
    list_display = ['page_name', 'visit_count', 'last_visit']
    list_editable = ['visit_count']
    readonly_fields = ['last_visit']
    list_filter = ['last_visit']
    search_fields = ['page_name']
