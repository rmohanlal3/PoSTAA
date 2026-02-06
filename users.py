"""
User management API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user_id, get_password_hash
from app.models.models import User, UserPreference
from app.models.schemas import (
    UserResponse,
    UserUpdate,
    UserPreferenceResponse,
    UserPreferenceUpdate,
    UserEngagementStats
)

router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Get current user profile"""
    user = db.query(User).filter(User.id == current_user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@router.patch("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Update current user profile"""
    user = db.query(User).filter(User.id == current_user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update fields
    if user_update.full_name is not None:
        user.full_name = user_update.full_name
    
    if user_update.password is not None:
        user.hashed_password = get_password_hash(user_update.password)
    
    db.commit()
    db.refresh(user)
    return user


@router.get("/preferences", response_model=UserPreferenceResponse)
async def get_user_preferences(
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Get user preferences"""
    preferences = db.query(UserPreference).filter(
        UserPreference.user_id == current_user_id
    ).first()
    
    if not preferences:
        # Create default preferences
        preferences = UserPreference(user_id=current_user_id)
        db.add(preferences)
        db.commit()
        db.refresh(preferences)
    
    return preferences


@router.put("/preferences", response_model=UserPreferenceResponse)
async def update_user_preferences(
    preferences_update: UserPreferenceUpdate,
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Update user preferences"""
    preferences = db.query(UserPreference).filter(
        UserPreference.user_id == current_user_id
    ).first()
    
    if not preferences:
        # Create new preferences
        preferences = UserPreference(
            user_id=current_user_id,
            **preferences_update.dict()
        )
        db.add(preferences)
    else:
        # Update existing preferences
        for field, value in preferences_update.dict(exclude_unset=True).items():
            setattr(preferences, field, value)
    
    db.commit()
    db.refresh(preferences)
    return preferences


@router.get("/stats", response_model=UserEngagementStats)
async def get_user_stats(
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Get user engagement statistics"""
    from sqlalchemy import func
    from app.models.models import ClipView, DailyClip, Theme
    
    # Get view statistics
    views = db.query(ClipView).filter(ClipView.user_id == current_user_id).all()
    
    total_views = len(views)
    total_watch_time = sum(v.watch_duration for v in views)
    completed_views = len([v for v in views if v.completed])
    completion_rate = (completed_views / total_views * 100) if total_views > 0 else 0
    
    # Get favorite themes
    theme_counts = db.query(
        Theme.name,
        func.count(ClipView.id).label('count')
    ).join(
        DailyClip, DailyClip.theme_id == Theme.id
    ).join(
        ClipView, ClipView.clip_id == DailyClip.id
    ).filter(
        ClipView.user_id == current_user_id
    ).group_by(
        Theme.name
    ).order_by(
        func.count(ClipView.id).desc()
    ).limit(5).all()
    
    favorite_themes = [theme.name for theme in theme_counts]
    
    return UserEngagementStats(
        total_views=total_views,
        total_watch_time=total_watch_time,
        completion_rate=completion_rate,
        favorite_themes=favorite_themes
    )


@router.delete("/me")
async def delete_account(
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Delete user account"""
    user = db.query(User).filter(User.id == current_user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # In production, you might want to soft delete or anonymize data
    db.delete(user)
    db.commit()
    
    return {"message": "Account deleted successfully"}
