#!/bin/bash

# 邮件配置修复脚本
# 此脚本将正确的邮件配置设置到环境变量中

echo "=== 修复邮件配置 ==="

# 正确的SMTP配置
export EMAIL_BACKEND='django.core.mail.backends.smtp.EmailBackend'
export EMAIL_HOST='smtp.qiye.aliyun.com'
export EMAIL_PORT=465
export EMAIL_USE_SSL='True'
export EMAIL_USE_TLS='False'
export EMAIL_HOST_USER='cvsp@mwxiao.com'
export EMAIL_HOST_PASSWORD='Mxiao123456'  # 请替换为真实的邮箱授权码
export DEFAULT_FROM_EMAIL='cvsp@mwxiao.com'
export EMAIL_TIMEOUT=30

echo "✅ 邮件配置已设置"
echo "   SMTP服务器: $EMAIL_HOST:$EMAIL_PORT"
echo "   发件邮箱: $EMAIL_HOST_USER"
echo "   使用SSL: $EMAIL_USE_SSL"

# 测试邮件发送
echo ""
echo "=== 开始测试邮件发送 ==="

cd /app/nmr_core_system
python test_email_configuration.py

echo ""
echo "=== 配置完成 ==="