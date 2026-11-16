from django.core.cache import cache
from django.utils import timezone
from django.db import models
from django.db.models import F, Sum
from .models import PageVisit

def visitor_counter(request):
    """
    Context processor to add visitor counter to all templates
    Uses PageVisit model for persistent storage
    """
    # Get current home visit count (without incrementing)
    home_visit, created = PageVisit.objects.get_or_create(
        page_name='home',
        defaults={'visit_count': 0}
    )
    
    # Only increment for unique visits using session
    if request.path == '/' or request.path == '':
        # Check if this session has already visited home page
        session_key = f'visited_home_{request.session.session_key}'
        if not cache.get(session_key):
            # Increment counter
            PageVisit.objects.filter(page_name='home').update(
                visit_count=F('visit_count') + 1
            )
            # Mark this session as visited
            cache.set(session_key, True, 3600)  # 1 hour
            # Refresh to get updated count
            home_visit.refresh_from_db()
    
    # Get total visits across all pages
    total_visits = PageVisit.objects.aggregate(
        total=Sum('visit_count')
    )['total'] or 0
    
    return {
        'total_visits': total_visits,
        'home_visits': home_visit.visit_count,
        # 用户显示名：优先 profile.display_name（可重复），否则登录用户名
        'user_display_name': _get_user_display_name(request),
    }


def _get_user_display_name(request):
    if not request.user.is_authenticated:
        return ''
    profile = getattr(request.user, 'profile', None)
    if profile and profile.display_name:
        return profile.display_name
    return request.user.username