"""邮箱真实存在性校验

原理: 查询域名 MX 记录 -> 连接收件方邮件服务器(SSL 465) -> RCPT TO 探测
- 163 等服务器会准确返回 250(存在) / 550(不存在)
- QQ / Gmail 等服务器不响应探测(超时),此时返回 None 表示"无法判定",
  不阻断注册(后续仍有验证邮件兜底)

注意: 探测有超时控制,最坏情况整个校验耗时约 ~15 秒。
"""
import logging
import socket
import smtplib
import ssl

import dns.resolver

logger = logging.getLogger(__name__)

# 整体探测超时(秒)
SMTP_PROBE_TIMEOUT = 10
DNS_TIMEOUT = 8


def verify_email_exists(email):
    """校验邮箱是否真实存在

    Returns:
        (True, '')            邮箱存在
        (False, reason)       邮箱不存在或域名无效(可直接拒绝注册)
        (None, reason)        服务器不支持探测(不阻断,由验证邮件兜底)
    """
    email = email.strip()
    if '@' not in email:
        return False, '邮箱格式不正确'

    domain = email.split('@')[-1].lower()

    # 1. 域名 MX 记录检查 —— 域名不存在直接拒绝
    try:
        mx_records = dns.resolver.resolve(domain, 'MX', lifetime=DNS_TIMEOUT)
        mx_host = str(sorted(mx_records, key=lambda r: r.preference)[0].exchange).rstrip('.')
    except dns.resolver.NXDOMAIN:
        return False, f'邮箱域名 {domain} 不存在，请检查拼写'
    except dns.resolver.NoAnswer:
        # 域名存在但无 MX 记录(如纯网站域名) -> 按RFC可退回A记录投递,
        # 但实际极少有邮箱在此类域名上,直接判不存在
        return False, f'邮箱域名 {domain} 不提供邮件服务，请确认地址无误'
    except Exception as e:
        logger.warning(f'MX 查询失败 {domain}: {e}')
        return None, f'邮件服务器暂时无法查询({type(e).__name__})，请稍后重试或继续提交'

    # 2. RCPT TO 探测
    old_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(SMTP_PROBE_TIMEOUT)
        ctx = ssl.create_default_context()
        s = smtplib.SMTP_SSL(mx_host, 465, timeout=SMTP_PROBE_TIMEOUT, context=ctx)
        try:
            s.helo('nmr-system-verify')
            s.mail('verify@nmr-check.local')
            code, msg = s.rcpt(email)
            if code == 250:
                return True, ''
            # 550 = 用户不存在; 551/553 等也视为拒收
            return False, f'邮箱 {email} 不存在，请确认地址无误'
        finally:
            try:
                s.quit()
            except Exception:
                pass
    except socket.timeout:
        return None, '邮件服务器未响应探测'
    except smtplib.SMTPServerDisconnected:
        return None, '邮件服务器拒绝探测'
    except Exception as e:
        logger.warning(f'RCPT 探测失败 {email}: {type(e).__name__}: {e}')
        return None, f'暂时无法验证邮箱({type(e).__name__})'
    finally:
        socket.setdefaulttimeout(old_timeout)
