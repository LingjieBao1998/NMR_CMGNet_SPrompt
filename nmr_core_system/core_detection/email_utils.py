"""
邮件发送工具模块
用于发送任务提交确认邮件和结果文件邮件
"""

from django.core.mail import EmailMessage, EmailMultiAlternatives
from django.template.loader import render_to_string
from django.conf import settings
from django.utils import timezone
import os
import logging

logger = logging.getLogger(__name__)


def send_task_confirmation_email(task):
    """
    发送任务提交确认邮件
    
    Args:
        task: NMRTask对象
    
    Returns:
        bool: 发送成功返回True，失败返回False
    """
    try:
        subject = f'任务提交成功 - {task.task_name}'
        context = {
            'task': task,
            'site_url': getattr(settings, 'SITE_URL', 'http://localhost:8000'),
            'submission_time': task.submitted_at.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # 渲染邮件内容
        text_content = render_to_string('core_detection/email/task_confirmation.txt', context)
        html_content = render_to_string('core_detection/email/task_confirmation.html', context)
        
        # 创建邮件
        email = EmailMultiAlternatives(
            subject=subject,
            body=text_content,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[task.email]
        )
        email.attach_alternative(html_content, "text/html")
        
        # 附加任务token文件
        if task.task_token and hasattr(task, 'get_token_file_path'):
            token_file_path = task.get_token_file_path()
            if os.path.exists(token_file_path):
                email.attach_file(token_file_path)
        
        # 发送邮件
        email.send()
        logger.info(f"任务确认邮件已发送至 {task.email}")
        return True
        
    except Exception as e:
        logger.error(f"发送任务确认邮件失败: {str(e)}")
        return False


def send_task_result_email(task):
    """
    发送任务完成结果邮件，包含结果文件
    
    Args:
        task: NMRTask对象
    
    Returns:
        bool: 发送成功返回True，失败返回False
    """
    try:
        if task.status != 'completed':
            logger.warning(f"任务 {task.task_id} 状态为 {task.status}，不发送结果邮件")
            return False
        
        subject = f'任务完成 - {task.task_name}'
        context = {
            'task': task,
            'site_url': getattr(settings, 'SITE_URL', 'http://localhost:8000'),
            'completion_time': task.processing_end_time.strftime('%Y-%m-%d %H:%M:%S') if task.processing_end_time else '未知',
            'processing_duration': f"{task.processing_duration:.2f}秒" if task.processing_duration else '未知',
        }
        
        # 渲染邮件内容
        text_content = render_to_string('core_detection/email/task_result.txt', context)
        html_content = render_to_string('core_detection/email/task_result.html', context)
        
        # 创建邮件
        email = EmailMultiAlternatives(
            subject=subject,
            body=text_content,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[task.email]
        )
        email.attach_alternative(html_content, "text/html")
        
        # 附加结果文件
        if task.output_file and os.path.exists(task.output_file.path):
            email.attach_file(task.output_file.path)
        
        # 附加token文件
        if task.task_token and hasattr(task, 'get_token_file_path'):
            token_file_path = task.get_token_file_path()
            if os.path.exists(token_file_path):
                email.attach_file(token_file_path)
        
        # 发送邮件
        email.send()
        logger.info(f"任务结果邮件已发送至 {task.email}")
        return True
        
    except Exception as e:
        logger.error(f"发送任务结果邮件失败: {str(e)}")
        return False


def send_task_failed_email(task):
    """
    发送任务失败通知邮件
    
    Args:
        task: NMRTask对象
    
    Returns:
        bool: 发送成功返回True，失败返回False
    """
    try:
        subject = f'任务失败 - {task.task_name}'
        context = {
            'task': task,
            'site_url': getattr(settings, 'SITE_URL', 'http://localhost:8000'),
            'error_message': task.error_message,
            'failure_time': task.processing_end_time.strftime('%Y-%m-%d %H:%M:%S') if task.processing_end_time else '未知',
        }
        
        # 渲染邮件内容
        text_content = render_to_string('core_detection/email/task_failed.txt', context)
        html_content = render_to_string('core_detection/email/task_failed.html', context)
        
        # 创建邮件
        email = EmailMultiAlternatives(
            subject=subject,
            body=text_content,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[task.email]
        )
        email.attach_alternative(html_content, "text/html")
        
        # 发送邮件
        email.send()
        logger.info(f"任务失败通知邮件已发送至 {task.email}")
        return True
        
    except Exception as e:
        logger.error(f"发送任务失败通知邮件失败: {str(e)}")
        return False


def send_custom_email(to_email, subject, message, attachments=None):
    """
    发送自定义邮件
    
    Args:
        to_email: 收件人邮箱
        subject: 邮件主题
        message: 邮件内容
        attachments: 附件列表（文件路径列表）
    
    Returns:
        bool: 发送成功返回True，失败返回False
    """
    try:
        email = EmailMessage(
            subject=subject,
            body=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[to_email]
        )
        
        # 添加附件
        if attachments:
            for attachment_path in attachments:
                if os.path.exists(attachment_path):
                    email.attach_file(attachment_path)
        
        # 发送邮件
        email.send()
        logger.info(f"自定义邮件已发送至 {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"发送自定义邮件失败: {str(e)}")
        return False


def send_contact_notification_email(contact_message):
    """
    发送联系表单通知邮件给管理员
    
    Args:
        contact_message: ContactMessage对象
    
    Returns:
        bool: 发送成功返回True，失败返回False
    """
    try:
        # 获取管理员邮箱列表
        admin_emails = getattr(settings, 'ADMIN_EMAILS', [settings.DEFAULT_FROM_EMAIL])
        
        subject = f'新的联系表单提交 - {contact_message.name}'
        context = {
            'message': contact_message,
            'site_url': getattr(settings, 'SITE_URL', 'http://localhost:8000'),
        }
        
        # 渲染邮件内容
        text_content = render_to_string('core_detection/email/contact_notification.txt', context)
        html_content = render_to_string('core_detection/email/contact_notification.html', context)
        
        # 创建邮件
        email = EmailMultiAlternatives(
            subject=subject,
            body=text_content,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=admin_emails
        )
        email.attach_alternative(html_content, "text/html")
        
        # 发送邮件
        email.send()
        logger.info(f"联系表单通知邮件已发送至 {admin_emails}")
        return True
        
    except Exception as e:
        logger.error(f"发送联系表单通知邮件失败: {str(e)}")
        return False