def app_identity(request):
    return {
        'current_username': request.session.get('username', 'Guest'),
        'is_logged_in': bool(request.session.get('is_logged_in')),
    }
