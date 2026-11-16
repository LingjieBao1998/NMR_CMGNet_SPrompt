from django.db import migrations


def archive_existing_tasks(apps, schema_editor):
    """把现有所有任务归入专用历史账号 historical_archive

    - 保留每个任务的 email/user_name 字段(原始提交者信息仍在任务记录里)
    - historical_archive 账号设为不可登录(is_active=False),仅作归档容器
    - 顺带为已有用户补建 UserProfile(历史账号标记为已批准)
    """
    User = apps.get_model('auth', 'User')
    UserProfile = apps.get_model('core_detection', 'UserProfile')
    NMRTask = apps.get_model('core_detection', 'NMRTask')
    User = apps.get_model('auth', 'User')

    archive_user, created = User.objects.get_or_create(
        username='historical_archive',
        defaults={
            'email': 'archive@nmr.local',
            'first_name': 'Historical',
            'last_name': 'Archive',
            'is_active': False,
        }
    )
    if not created and archive_user.is_active:
        archive_user.is_active = False
        archive_user.save(update_fields=['is_active'])

    NMRTask.objects.exclude(submitted_by=archive_user).update(submitted_by=archive_user)

    # 为现存所有用户补建 profile；历史账号直接视为已批准可用
    for user in User.objects.all():
        UserProfile.objects.get_or_create(
            user=user,
            defaults={
                'email_verified': True,
                'approval_status': 'approved' if not user.is_active else 'pending',
            }
        )


def unarchive(apps, schema_editor):
    """回滚：删除归档账号(任务会因 CASCADE 删除——所以回滚前人工处理)

    注意：此反向操作会级联删除归档任务，仅用于开发环境回滚。
    """
    User = apps.get_model('auth', 'User')
    User.objects.filter(username='historical_archive').delete()


class Migration(migrations.Migration):

    dependencies = [
        ('core_detection', '0010_userprofile'),
    ]

    operations = [
        migrations.RunPython(archive_existing_tasks, unarchive),
    ]
