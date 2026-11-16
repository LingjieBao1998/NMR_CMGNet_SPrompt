"""用户注册/登录/邮箱验证视图

流程：
1. register: 创建用户 + UserProfile，发验证邮件(失败则回滚)
2. verify_email: 校验 token -> 账号激活，自动登录并跳转主页
3. login: 邮箱或用户名登录
"""
import logging
import secrets
from datetime import timedelta

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from .email_utils import send_custom_email
from .email_verify import verify_email_exists
from .models import UserProfile

logger = logging.getLogger(__name__)

# 验证邮件有效期（小时）
EMAIL_TOKEN_VALID_HOURS = 48


def _send_verification_email(user, profile):
    """发送邮箱验证邮件，返回是否成功"""
    token = secrets.token_urlsafe(32)
    profile.email_verification_token = token
    profile.save(update_fields=['email_verification_token'])

    from django.conf import settings
    site_url = settings.SITE_URL.rstrip('/')
    verify_url = f"{site_url}/verify-email/{token}/"

    subject = "邮箱验证 - NMR核心结构检测系统"
    body = (
        f"您好，\n\n"
        f"感谢注册 NMR 核心结构智慧检测系统。\n\n"
        f"请点击以下链接完成邮箱验证并激活账号（{EMAIL_TOKEN_VALID_HOURS} 小时内有效）：\n"
        f"{verify_url}\n\n"
        f"点击链接后您的账号将自动激活并登录，即可开始使用系统。\n\n"
        f"如果这不是您本人的操作，请忽略本邮件。\n"
    )
    return send_custom_email(user.email, subject, body)


@require_http_methods(['GET', 'POST'])
def register_view(request):
    """用户注册：创建账号并发送验证邮件"""
    if request.user.is_authenticated:
        return redirect('core_detection:home')

    if request.method == 'POST':
        display_name = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        institution = request.POST.get('institution', '').strip()
        password1 = request.POST.get('password1', '')
        password2 = request.POST.get('password2', '')

        errors = []
        if not display_name:
            errors.append('请输入用户名（显示名，可与他人重复）')
        if not email:
            errors.append('请输入邮箱')
        elif User.objects.filter(email__iexact=email).exists():
            errors.append('该邮箱已被注册')
        if not password1 or len(password1) < 8:
            errors.append('密码至少 8 位')
        if password1 != password2:
            errors.append('两次输入的密码不一致')

        # 邮箱真实存在性校验（域名+用户名）。仅在格式和唯一性都通过后进行，
        # 避免对明显无效的输入浪费探测时间。
        if not errors:
            exists, reason = verify_email_exists(email)
            if exists is False:
                errors.append(reason or '邮箱不存在，请确认地址无误')

        if errors:
            for e in errors:
                messages.error(request, e)
            return render(request, 'core_detection/register.html', {
                'page_title': 'Register',
                'form_data': {'username': display_name, 'email': email, 'institution': institution},
            })

        # 用户名仅作显示名可重复；登录账号内部从邮箱生成唯一 username
        base = email.split('@')[0].lower()
        candidate = base
        suffix = 1
        while User.objects.filter(username__iexact=candidate).exists():
            suffix += 1
            candidate = f"{base}{suffix}"

        try:
            with transaction.atomic():
                user = User.objects.create_user(
                    username=candidate,
                    email=email,
                    password=password1,
                    is_active=True,  # 验证邮件中的链接完成激活并自动登录
                )
                profile = UserProfile.objects.create(
                    user=user,
                    display_name=display_name,
                    institution=institution,
                    approval_status='pending',  # 邮箱验证链接点击后置为 approved
                )
                # 在事务内发信，失败则整体回滚（用户不会注册成功却收不到邮件）
                if not _send_verification_email(user, profile):
                    raise RuntimeError('验证邮件发送失败')
        except RuntimeError:
            messages.error(request, '验证邮件发送失败，请联系管理员或稍后重试')
            return render(request, 'core_detection/register.html', {
                'page_title': 'Register',
                'form_data': {'username': display_name, 'email': email, 'institution': institution},
            })
        except Exception:
            logger.exception('注册失败')
            messages.error(request, '注册失败，请稍后重试')
            return render(request, 'core_detection/register.html', {'page_title': 'Register'})

        logger.info(f"新用户注册: {display_name} <{email}> (登录名 {user.username})，等待邮箱验证")
        return render(request, 'core_detection/email_verify_notice.html', {
            'page_title': 'Verify Email',
            'email': email,
        })

    return render(request, 'core_detection/register.html', {'page_title': 'Register'})


@require_http_methods(['GET'])
def verify_email_view(request, token):
    """邮箱验证：校验 token，激活账号并自动登录"""
    profile = UserProfile.objects.select_related('user').filter(
        email_verification_token=token
    ).first()

    if profile is None:
        messages.error(request, '无效的验证链接')
        return redirect('core_detection:register')

    if profile.email_verified:
        # 已验证过：如果未登录则直接登录，已登录回主页
        if not request.user.is_authenticated:
            login(request, profile.user, backend='django.contrib.auth.backends.ModelBackend')
        messages.info(request, '邮箱已验证，欢迎回来！')
        return redirect('core_detection:home')

    # 校验时效
    deadline = profile.registered_at + timedelta(hours=EMAIL_TOKEN_VALID_HOURS)
    if timezone.now() > deadline:
        messages.error(request, '验证链接已过期，请重新注册')
        return redirect('core_detection:register')

    # 激活：邮箱验证通过即视为完全可用（无审批环节）
    profile.email_verified = True
    profile.email_verification_token = None  # 一次性使用
    profile.approval_status = 'approved'
    profile.approved_at = timezone.now()
    profile.save(update_fields=['email_verified', 'email_verification_token',
                                'approval_status', 'approved_at'])

    user = profile.user
    if not user.is_active:
        user.is_active = True
        user.save(update_fields=['is_active'])

    # 自动登录并跳转主页
    login(request, user, backend='django.contrib.auth.backends.ModelBackend')
    display = profile.display_name or user.username
    messages.success(request, f'邮箱验证成功，欢迎加入，{display}！')
    logger.info(f"用户 {user.username} 邮箱验证完成并自动登录")

    return redirect('core_detection:home')


@require_http_methods(['GET', 'POST'])
def login_view(request):
    """登录：未审批账号即使密码正确也拒绝"""
    if request.user.is_authenticated:
        return redirect('core_detection:home')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')

        # 支持用户名或邮箱登录：admin 账号仅允许用户名登录
        if '@' in username and username.split('@')[0].lower() != 'admin':
            user = (User.objects.filter(email__iexact=username).first() or
                    User.objects.filter(username__iexact=username).first())
            if user is not None and user.check_password(password):
                pass  # 密码正确，用该用户继续
            else:
                user = None
        else:
            user = authenticate(request, username=username, password=password)

        if user is None:
            messages.error(request, '用户名或密码错误')
        else:
            profile = getattr(user, 'profile', None)
            if profile and not profile.email_verified:
                messages.error(request, '邮箱尚未验证，请查收验证邮件并点击其中的激活链接')
            else:
                login(request, user)
                display = (profile.display_name if profile and profile.display_name else None) or user.username
                messages.success(request, f'欢迎回来，{display}！')
                next_url = request.POST.get('next') or request.GET.get('next')
                if next_url and next_url.startswith('/'):
                    return redirect(next_url)
                return redirect('core_detection:home')

    return render(request, 'core_detection/login.html', {
        'page_title': 'Login',
        'next': request.GET.get('next', ''),
    })


@login_required
@require_http_methods(['POST'])
def logout_view(request):
    """登出"""
    logout(request)
    messages.info(request, '您已退出登录')
    return redirect('core_detection:home')


@require_http_methods(['GET'])
def api_check_email(request):
    """AJAX 邮箱校验：格式 -> 是否已注册 -> 是否真实存在"""
    import json as _json
    from django.core.validators import validate_email as dj_validate
    from django.core.exceptions import ValidationError as DjValidationError

    email = request.GET.get('email', '').strip()
    if not email:
        return JsonResponse({'available': False, 'reason': '请输入邮箱'})

    try:
        dj_validate(email)
    except DjValidationError:
        return JsonResponse({'available': False, 'reason': '邮箱格式不正确'})

    if User.objects.filter(email__iexact=email).exists():
        return JsonResponse({'available': False, 'reason': '该邮箱已被注册'})

    exists, reason = verify_email_exists(email)
    if exists is False:
        return JsonResponse({'available': False, 'reason': reason or '邮箱不存在，请确认地址无误'})
    if exists is None:
        # 无法探测(QQ/Gmail等)，不阻断
        return JsonResponse({'available': True, 'verified': False,
                             'reason': reason or '暂时无法在线验证，提交后将发送验证邮件确认'})

    return JsonResponse({'available': True, 'verified': True})


# ==================== 忘记密码 ====================

# 密码重置链接有效期（小时）
PASSWORD_RESET_VALID_HOURS = 1


def _send_password_reset_email(user, profile):
    """发送密码重置邮件，返回是否成功"""
    token = secrets.token_urlsafe(32)
    profile.password_reset_token = token
    profile.password_reset_token_at = timezone.now()
    profile.save(update_fields=['password_reset_token', 'password_reset_token_at'])

    from django.conf import settings
    site_url = settings.SITE_URL.rstrip('/')
    reset_url = f"{site_url}/reset-password/{token}/"

    subject = "重置密码 - NMR核心结构检测系统"
    body = (
        f"您好，\n\n"
        f"我们收到了重置您账号密码的请求。\n\n"
        f"请点击以下链接设置新密码（{PASSWORD_RESET_VALID_HOURS} 小时内有效，一次性使用）：\n"
        f"{reset_url}\n\n"
        f"如果这不是您本人的操作，请忽略本邮件，您的密码不会发生变化。\n"
    )
    return send_custom_email(user.email, subject, body)


@require_http_methods(['GET', 'POST'])
def forgot_password_view(request):
    """忘记密码：输入邮箱，发送重置链接

    无论邮箱是否注册都显示相同的成功提示（不泄露哪些邮箱已注册），
   但只在邮箱存在时真正发信。
    """
    if request.method == 'POST':
        email = request.POST.get('email', '').strip()

        if not email:
            messages.error(request, '请输入邮箱')
        else:
            sent = False
            user = User.objects.filter(email__iexact=email).select_related('profile').first()
            if user is not None:
                profile = getattr(user, 'profile', None)
                if profile is None:
                    from .models import UserProfile as _UP
                    profile = _UP.objects.create(user=user)
                # 邮箱未验证的账号也允许重置（重置后视为已验证，因为掌握了邮箱控制权）
                sent = _send_password_reset_email(user, profile)
                if sent:
                    logger.info(f"密码重置邮件已发送: {user.username}")
                else:
                    logger.error(f"密码重置邮件发送失败: {email}")
            # 邮箱不存在时静默，统一提示，防止枚举注册邮箱

        return render(request, 'core_detection/forgot_password_sent.html', {
            'page_title': 'Reset Email Sent',
            'email': email,
        })

    return render(request, 'core_detection/forgot_password.html', {'page_title': 'Forgot Password'})


@require_http_methods(['GET', 'POST'])
def reset_password_view(request, token):
    """重置密码：校验 token，设置新密码并自动登录"""
    profile = UserProfile.objects.select_related('user').filter(
        password_reset_token=token
    ).first()

    if profile is None or not profile.password_reset_token_at:
        messages.error(request, '无效的重置链接')
        return redirect('core_detection:login')

    # 校验时效
    deadline = profile.password_reset_token_at + timedelta(hours=PASSWORD_RESET_VALID_HOURS)
    if timezone.now() > deadline:
        profile.password_reset_token = None
        profile.password_reset_token_at = None
        profile.save(update_fields=['password_reset_token', 'password_reset_token_at'])
        messages.error(request, '重置链接已过期，请重新申请')
        return redirect('core_detection:forgot_password')

    user = profile.user

    if request.method == 'POST':
        password1 = request.POST.get('password1', '')
        password2 = request.POST.get('password2', '')

        errors = []
        if not password1 or len(password1) < 8:
            errors.append('密码至少 8 位')
        if password1 != password2:
            errors.append('两次输入的密码不一致')

        if errors:
            for e in errors:
                messages.error(request, e)
            return render(request, 'core_detection/reset_password.html', {
                'page_title': 'Reset Password', 'token': token})

        user.set_password(password1)
        user.save()

        # token 一次性使用；掌握邮箱即视为已验证
        profile.password_reset_token = None
        profile.password_reset_token_at = None
        if not profile.email_verified:
            profile.email_verified = True
        profile.save(update_fields=['password_reset_token', 'password_reset_token_at', 'email_verified'])

        # 重置成功后自动登录
        login(request, user, backend='django.contrib.auth.backends.ModelBackend')
        display = profile.display_name or user.username
        messages.success(request, f'密码已重置，欢迎回来，{display}！')
        logger.info(f"用户 {user.username} 通过邮件链接重置密码并登录")
        return redirect('core_detection:home')

    return render(request, 'core_detection/reset_password.html', {
        'page_title': 'Reset Password', 'token': token})
