"""
Smart Alert Automation for Industrial Maintenance Intelligence
Provides SMS and WhatsApp notifications via Twilio for critical equipment failures
"""

import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime 
from dotenv import load_dotenv  
load_dotenv()
import google.generativeai as genai
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("⚠️ Twilio not installed. Run: pip install twilio")
def summarize_for_whatsapp(ai_analysis: str, machine_id: str, severity: str) -> str:
    """
    Use Gemini to create a concise, action-focused summary under 800 chars
    
    Args:
        ai_analysis: Full AI diagnostic text
        machine_id: Machine identifier
        severity: CRITICAL or WARNING
    
    Returns:
        Summarized text focused on problem-solving
    """
    try:
        # Get API key from environment
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            return "⚠️ AI summary unavailable. Check full report in dashboard."
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Create focused prompt
        prompt = f"""You are a maintenance engineer communicating critical equipment issues via WhatsApp.

ORIGINAL AI ANALYSIS:
{ai_analysis}

TASK: Create an ultra-concise summary (MAX 800 characters) for {machine_id} - {severity} alert.

FOCUS ON:
1. ROOT CAUSE (1-2 sentences max)
2. IMMEDIATE ACTION REQUIRED (bullet points, very specific)
3. TIMELINE (when to act)
4. CONSEQUENCE if ignored (1 sentence)

RULES:
- Be direct and actionable
- Use technical terms (vibration, thermal, efficiency)
- NO pleasantries or filler
- Prioritize what technician must DO NOW
- Keep under 800 chars total

Format:
🔍 CAUSE: [brief root cause]
⚡ ACTION: [specific steps]
⏰ TIMELINE: [urgency]
⚠️ RISK: [consequence]
"""
        
        response = model.generate_content(prompt)
        summary = response.text.strip()
        
        # Safety check - if still too long, truncate intelligently
        if len(summary) > 800:
            summary = summary[:780] + "...\n[Full report in dashboard]"
        
        return summary
        
    except Exception as e:
        print(f"⚠️ Gemini summarization failed: {e}")
        # Fallback: ultra-short summary
        return f"⚠️ {severity} issue detected. Action required. See dashboard for details."

class AlertService:
    """Handles SMS and WhatsApp alerts via Twilio"""
    
    def __init__(self):
        """Initialize Twilio client from environment variables"""
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_phone = os.getenv("TWILIO_PHONE_NUMBER")
        self.to_phone = os.getenv("ALERT_RECIPIENT_PHONE")  # e.g., +919876543210
        
        self.client = None
        self.is_configured = False
        
        if TWILIO_AVAILABLE and all([self.account_sid, self.auth_token, self.from_phone, self.to_phone]):
            try:
                self.client = Client(self.account_sid, self.auth_token)
                self.is_configured = True
            except Exception as e:
                print(f"⚠️ Twilio configuration error: {e}")
                self.is_configured = False
        else:
            missing = []
            if not self.account_sid:
                missing.append("TWILIO_ACCOUNT_SID")
            if not self.auth_token:
                missing.append("TWILIO_AUTH_TOKEN")
            if not self.from_phone:
                missing.append("TWILIO_PHONE_NUMBER")
            if not self.to_phone:
                missing.append("ALERT_RECIPIENT_PHONE")
            
            if missing:
                print(f"⚠️ Missing Twilio credentials: {', '.join(missing)}")
    
    def send_sms(self, message: str) -> bool:
        """
        Send SMS alert
        
        Args:
            message: One-line alert message
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_configured:
            print("⚠️ SMS not sent: Twilio not configured")
            return False
        
        try:
            self.client.messages.create(
                body=message,
                from_=self.from_phone,
                to=self.to_phone
            )
            print(f"✅ SMS sent successfully to {self.to_phone}")
            return True
        except Exception as e:
            print(f"❌ SMS failed: {e}")
            return False
    
    def send_whatsapp(self, message: str) -> bool:
        """
        Send WhatsApp alert via Twilio Sandbox
        
        Args:
            message: Detailed alert message (can be multi-line)
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_configured:
            print("⚠️ WhatsApp not sent: Twilio not configured")
            return False
        
        try:
            # Twilio WhatsApp Sandbox number
            whatsapp_from = "whatsapp:+14155238886"
            whatsapp_to = f"whatsapp:{self.to_phone}"
            
            self.client.messages.create(
                body=message,
                from_=whatsapp_from,
                to=whatsapp_to
            )
            print(f"✅ WhatsApp sent successfully to {self.to_phone}")
            return True
        except Exception as e:
            print(f"❌ WhatsApp failed: {e}")
            return False


# Global alert service instance
_alert_service = AlertService()


def check_alert_conditions(prediction_row: Dict[str, Any]) -> Optional[str]:
    """
    Evaluate alert conditions based on prediction data
    
    Args:
        prediction_row: Dictionary with prediction values (efficiency_index, vibration_index, thermal_index)
    
    Returns:
        "CRITICAL", "WARNING", or None
    """
    efficiency = prediction_row.get('efficiency_index', 100)
    vibration = prediction_row.get('vibration_index', 0)
    thermal = prediction_row.get('thermal_index', 0)
    
    # CRITICAL conditions
    if efficiency < 40 or vibration > 80 or thermal > 80:
        return "CRITICAL"
    
    # WARNING conditions
    if efficiency < 65:
        return "WARNING"
    
    return None


def send_sms_alert(machine_id: str, severity: str, prediction_row: Dict[str, Any]) -> bool:
    """
    Send one-line SMS alert
    
    Args:
        machine_id: Machine identifier
        severity: "CRITICAL" or "WARNING"
        prediction_row: Prediction data
    
    Returns:
        True if sent successfully
    """
    efficiency = prediction_row.get('efficiency_index', 0)
    vibration = prediction_row.get('vibration_index', 0)
    thermal = prediction_row.get('thermal_index', 0)
    
    # One-line high-signal message
    if severity == "CRITICAL":
        message = f"🚨 CRITICAL: {machine_id} - Eff:{efficiency:.0f}% Vib:{vibration:.0f} Temp:{thermal:.0f} - IMMEDIATE ACTION REQUIRED"
    else:
        message = f"⚠️ WARNING: {machine_id} - Efficiency at {efficiency:.0f}% - Maintenance recommended"
    
    return _alert_service.send_sms(message)


def send_whatsapp_alert(
    machine_row: Dict[str, Any],
    prediction_row: Dict[str, Any],
    ai_analysis: Optional[str],
    severity: str
) -> bool:
    """
    Send detailed WhatsApp alert with AI analysis (under 1600 char limit)
    
    Args:
        machine_row: Original machine sensor data
        prediction_row: ML prediction outputs
        ai_analysis: Full AI diagnostic text from Gemini
        severity: "CRITICAL" or "WARNING"
    
    Returns:
        True if sent successfully
    """
    if not _alert_service.is_configured:
        print("⚠️ WhatsApp not sent: Twilio not configured")
        return False
    
    machine_id = machine_row.get('machine_id', 'UNKNOWN')
    machine_type = machine_row.get('machine_type', 'Unknown Type')
    
    efficiency = prediction_row.get('efficiency_index', 0)
    vibration = prediction_row.get('vibration_index', 0)
    thermal = prediction_row.get('thermal_index', 0)
    health_score = prediction_row.get('health_score', 0)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # Build header (compact)
    severity_emoji = "🚨" if severity == "CRITICAL" else "⚠️"
    
    # Header section (~200 chars)
    message = f"""{severity_emoji} *{severity}*
Asset: {machine_id} ({machine_type})
Time: {timestamp}

*Metrics:*
Health: {health_score:.0f} | Eff: {efficiency:.0f}%
Vib: {vibration:.0f} | Thermal: {thermal:.0f}

"""
    
    # AI Analysis section (max ~800 chars after summarization)
    if ai_analysis:
        # Use Gemini to summarize
        ai_summary = summarize_for_whatsapp(ai_analysis, machine_id, severity)
        message += f"*AI DIAGNOSIS:*\n{ai_summary}\n\n"
    else:
        message += "⚠️ AI analysis pending. Check dashboard.\n\n"
    
    # Footer section (~100 chars)
    message += "━━━━━━━━━━━━━━━━━━\n"
    
    if severity == "CRITICAL":
        message += "🔴 *ACT NOW* - Emergency maintenance required"
    else:
        message += "🟡 *SCHEDULE* - Plan maintenance within 48h"
    
    # Final safety check - should be well under 1600 now
    if len(message) > 1580:
        # Emergency truncation (rare case)
        message = message[:1550] + "\n...\n[See full report in dashboard]"
    
    try:
        # Twilio WhatsApp Sandbox number
        whatsapp_from = "whatsapp:+14155238886"
        whatsapp_to = f"whatsapp:{_alert_service.to_phone}"
        
        _alert_service.client.messages.create(
            body=message,
            from_=whatsapp_from,
            to=whatsapp_to
        )
        print(f"✅ WhatsApp sent successfully to {_alert_service.to_phone}")
        return True
    except Exception as e:
        print(f"❌ WhatsApp failed: {e}")
        return False


def trigger_alerts(
    machine_row: Dict[str, Any],
    prediction_row: Dict[str, Any],
    ai_analysis: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point for alert system
    Evaluates conditions and sends appropriate notifications
    
    Args:
        machine_row: Original machine sensor data
        prediction_row: ML prediction outputs
        ai_analysis: Full AI diagnostic text (optional)
    
    Returns:
        Dictionary with alert status:
        {
            'alert_triggered': bool,
            'severity': str or None,
            'sms_sent': bool,
            'whatsapp_sent': bool,
            'message': str
        }
    """
    result = {
        'alert_triggered': False,
        'severity': None,
        'sms_sent': False,
        'whatsapp_sent': False,
        'message': 'No alerts triggered'
    }
    
    # Check if alerts are needed
    severity = check_alert_conditions(prediction_row)
    
    if severity is None:
        result['message'] = 'Asset operating normally - no alerts needed'
        return result
    
    # Alerts are needed
    result['alert_triggered'] = True
    result['severity'] = severity
    
    machine_id = machine_row.get('machine_id', 'UNKNOWN')
    
    # Send SMS (always for any alert)
    sms_success = send_sms_alert(machine_id, severity, prediction_row)
    result['sms_sent'] = sms_success
    
    # Send WhatsApp (detailed analysis for CRITICAL only, or if AI analysis available)
    if severity == "CRITICAL" or ai_analysis:
        whatsapp_success = send_whatsapp_alert(machine_row, prediction_row, ai_analysis, severity)
        result['whatsapp_sent'] = whatsapp_success
    
    # Build result message
    alerts_sent = []
    if result['sms_sent']:
        alerts_sent.append('SMS')
    if result['whatsapp_sent']:
        alerts_sent.append('WhatsApp')
    
    if alerts_sent:
        result['message'] = f"{severity} alert sent via {' and '.join(alerts_sent)}"
    else:
        result['message'] = f"{severity} alert triggered but sending failed (check Twilio config)"
    
    return result


# Configuration check utility
def is_configured() -> bool:
    """Check if alert service is properly configured"""
    return _alert_service.is_configured


def get_configuration_status() -> Dict[str, Any]:
    """Get detailed configuration status for debugging"""
    return {
        'twilio_available': TWILIO_AVAILABLE,
        'is_configured': _alert_service.is_configured,
        'has_account_sid': bool(_alert_service.account_sid),
        'has_auth_token': bool(_alert_service.auth_token),
        'has_from_phone': bool(_alert_service.from_phone),
        'has_to_phone': bool(_alert_service.to_phone),
        'from_phone': _alert_service.from_phone if _alert_service.from_phone else 'Not set',
        'to_phone': _alert_service.to_phone if _alert_service.to_phone else 'Not set'
    }