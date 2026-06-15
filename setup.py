#!/usr/bin/env python3
"""
Context Engine Magical Onboarding Script

Creates the 30-second conversational setup experience that makes
Context Engine irresistible.
"""

import os
import json
import time
import subprocess
import sys
from pathlib import Path

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header():
    print(f"""
{Colors.BLUE}┌─────────────────────────────────────────────────┐
│  Welcome to Context Engine                      │
│  Personal AI memory that never forgets          │
└─────────────────────────────────────────────────┘{Colors.END}

I'll set up persistent memory for your AI sessions.
This means no more re-explaining context, no more "what were we working on yesterday?"

First, let me get to know you:
""")

def get_user_input():
    """Collect the 3 essential pieces of information"""
    
    # Question 1: What should I call you?
    name = input(f"  {Colors.BOLD}What should I call you?{Colors.END} ").strip()
    if not name:
        name = "User"  # Fallback
    
    print()
    
    # Question 2: What's your role?
    role = input(f"  {Colors.BOLD}What's your role?{Colors.END} (Engineer, PM, Designer, etc.) ").strip()
    if not role:
        role = "Team Member"  # Fallback
    
    print()
    
    # Question 3: Communication style
    print(f"  {Colors.BOLD}How should your AI assistant talk to you?{Colors.END}")
    print(f"  a) Direct and concise")
    print(f"  b) Warm and conversational") 
    print(f"  c) Detailed and thorough")
    
    style_choice = input(f"  → ").strip().lower()
    
    style_map = {
        'a': 'direct and concise',
        'b': 'warm and conversational', 
        'c': 'detailed and thorough'
    }
    
    communication_style = style_map.get(style_choice, 'warm and conversational')
    
    return {
        'name': name,
        'role': role,
        'communication_style': communication_style
    }

def generate_soul_md(user_info):
    """Generate personalized SOUL.md for the AI assistant"""
    
    style_instructions = {
        'direct and concise': "Be direct, efficient, and to-the-point. No unnecessary preamble or verbose explanations.",
        'warm and conversational': "Be warm, friendly, and conversational. Use a natural, caring tone while remaining helpful.",
        'detailed and thorough': "Provide comprehensive, detailed explanations. Include context, reasoning, and multiple perspectives."
    }
    
    soul_content = f"""# AI Assistant Identity

## Who I Am
I am {user_info['name']}'s personal AI assistant with persistent memory across all sessions.

## My Voice & Communication Style
{style_instructions[user_info['communication_style']]}

I remember everything from our previous conversations. I know {user_info['name']} is a {user_info['role']} and I adapt my responses accordingly.

## Core Principles
- I remember context from previous sessions automatically
- I reference relevant past work naturally in conversation
- I never make {user_info['name']} re-explain things I should remember
- I build on our shared history to be genuinely helpful

## Memory Philosophy
I don't announce what I remember - I just use it naturally. When {user_info['name']} mentions "that API issue," I know which one. When they say "the deployment script," I remember the context.

This is what makes me different from other AI assistants. I never forget.
"""
    
    return soul_content

def generate_user_md(user_info):
    """Generate personalized USER.md profile"""
    
    user_content = f"""# User Profile: {user_info['name']}

## Identity
- Name: {user_info['name']}
- Role: {user_info['role']}
- Communication Preference: {user_info['communication_style']}

## Context
This user profile was created during Context Engine setup on {time.strftime('%Y-%m-%d')}.

The AI assistant will learn more about {user_info['name']}'s work patterns, preferences, and context through natural conversation over time.

## Memory Integration
This profile integrates with Context Engine to provide persistent memory across AI sessions. The assistant remembers previous conversations, projects, decisions, and context automatically.
"""
    
    return user_content

def run_setup_steps(user_info):
    """Execute all the automated setup steps"""
    
    print(f"\n{Colors.GREEN}Perfect! Setting up your personal AI memory...{Colors.END}\n")
    
    steps = [
        ("Checking Python environment...", check_python),
        ("Installing dependencies...", install_dependencies), 
        ("Creating memory vault at ~/.context-engine/", create_directories),
        ("Detecting Claude Code installation...", detect_claude_code),
        ("Installing memory hooks...", install_hooks),
        ("Starting Context Engine server...", start_server),
        ("Generating your AI assistant's identity...", lambda: generate_identity_files(user_info))
    ]
    
    for step_name, step_func in steps:
        print(f"✓ {step_name}")
        try:
            step_func()
            time.sleep(0.5)  # Brief pause for visual effect
        except Exception as e:
            print(f"  {Colors.RED}Error: {e}{Colors.END}")
            return False
    
    return True

def check_python():
    """Verify Python 3.9+ is available"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        raise Exception("Python 3.9+ required")

def install_dependencies():
    """Install required packages"""
    required_packages = ["flask", "flask-caching", "pyyaml", "httpx", "anthropic"]
    
    for package in required_packages:
        subprocess.run([sys.executable, "-m", "pip", "install", package], 
                      capture_output=True, check=True)

def create_directories():
    """Create Context Engine directory structure"""
    ce_dir = Path.home() / ".context-engine"
    ce_dir.mkdir(exist_ok=True)
    (ce_dir / "cursors").mkdir(exist_ok=True)

def detect_claude_code():
    """Check if Claude Code is installed"""
    claude_dir = Path.home() / ".claude"
    if not claude_dir.exists():
        raise Exception("Claude Code not found. Please install Claude Code first.")

def install_hooks():
    """Install Context Engine hooks into Claude Code"""
    ce_repo = Path(__file__).parent.resolve()
    hook_src = ce_repo / "hooks"
    if not hook_src.exists():
        raise Exception(f"Hook source not found at {hook_src}")

    hook_dst = Path.home() / ".claude" / "plugins" / "local" / "context-engine" / "hooks"
    hook_dst.mkdir(parents=True, exist_ok=True)

    import shutil
    for fname in ("connector.py", "hooks.json"):
        src = hook_src / fname
        if src.exists():
            shutil.copy2(src, hook_dst / fname)

    (hook_dst / "connector.py").chmod(0o755)

    # Wire hooks into settings.json (Claude Code reads hooks from here, not from plugins dir)
    settings_path = Path.home() / ".claude" / "settings.json"
    connector_path = str(hook_dst / "connector.py")

    settings = {}
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())

    hooks = settings.setdefault("hooks", {})
    hook_entries = {
        "SessionStart": f"python3 {connector_path} session-start",
        "UserPromptSubmit": f"python3 {connector_path} user-prompt",
        "Stop": f"python3 {connector_path} stop",
    }
    timeouts = {"SessionStart": 5, "UserPromptSubmit": 3, "Stop": 5}

    for event, command in hook_entries.items():
        existing = hooks.get(event, [])
        already_wired = any(
            "connector.py" in h.get("command", "")
            for entry in existing
            for h in (entry.get("hooks", []) if isinstance(entry, dict) else [])
        )
        if not already_wired:
            existing.append({
                "hooks": [{
                    "type": "command",
                    "command": command,
                    "timeout": timeouts[event],
                }]
            })
            hooks[event] = existing

    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

def start_server():
    """Start the Context Engine server and install LaunchAgent for auto-start"""
    ce_repo = Path(__file__).parent.resolve()

    # Create LaunchAgent for auto-start on login (macOS)
    if sys.platform == "darwin":
        plist_dir = Path.home() / "Library" / "LaunchAgents"
        plist_dir.mkdir(parents=True, exist_ok=True)
        plist_path = plist_dir / "com.context-engine.server.plist"
        python_path = sys.executable
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.context-engine.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>context_engine.server</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>{ce_repo}</string>
    <key>StandardOutPath</key>
    <string>/tmp/ce_server.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/ce_server.log</string>
</dict>
</plist>"""
        plist_path.write_text(plist_content)

        # Load the LaunchAgent into the current session (starts the server now)
        subprocess.run(["launchctl", "unload", str(plist_path)],
                       capture_output=True)
        subprocess.run(["launchctl", "load", str(plist_path)],
                       capture_output=True, check=True)
    else:
        # Non-macOS: start directly
        subprocess.Popen(
            [sys.executable, "-m", "context_engine.server"],
            cwd=str(ce_repo),
            stdout=open("/tmp/ce_server.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    # Verify server is up
    import urllib.request
    for _ in range(10):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8850/context/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise Exception("Server failed to start — check /tmp/ce_server.log")

def generate_identity_files(user_info):
    """Create SOUL.md, USER.md, and config.yaml"""
    ce_dir = Path.home() / ".context-engine"

    soul_path = ce_dir / "SOUL.md"
    user_path = ce_dir / "USER.md"

    soul_path.write_text(generate_soul_md(user_info))
    user_path.write_text(generate_user_md(user_info))

    # Write config.yaml with identity paths
    config_path = ce_dir / "config.yaml"
    if not config_path.exists():
        config_content = f"""identity:
  soul: {soul_path}
  user_model: {user_path}

server:
  port: 8850
  host: 127.0.0.1

confidence:
  decay_half_life_days: 14
  archive_threshold: 0.1

memory:
  capacity_threshold: 0.8
  aging_days: 30
"""
        config_path.write_text(config_content)

def print_success():
    """Show the completion message"""
    print(f"""
{Colors.GREEN}{Colors.BOLD}🎉 Context Engine is live!{Colors.END}

Your AI now has persistent memory across all sessions.
Start a new Claude Code conversation to see the magic.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{Colors.BOLD}What we just built for you:{Colors.END}

🧠 {Colors.BOLD}Personal Knowledge Graph{Colors.END}
   A local database that captures decisions, insights, and
   context from every AI conversation you have

📚 {Colors.BOLD}Working Memory System{Colors.END}
   Your AI remembers what you worked on yesterday, last week,
   and can connect patterns across all your sessions

🎭 {Colors.BOLD}AI Identity & Relationship{Colors.END}
   Your assistant knows who you are, how you prefer to
   communicate, and will adapt to work better with you over time

🔗 {Colors.BOLD}Seamless Integration{Colors.END}
   Everything happens invisibly — no new apps to learn,
   just AI that actually remembers

🏠 {Colors.BOLD}Privacy First{Colors.END}
   All your data stays on your machine. You control what
   gets shared (if anything). Your memory belongs to you.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{Colors.BLUE}What happens next:{Colors.END}
• Your AI will remember this conversation
• Future sessions will reference past work naturally
• The magic grows with each session

{Colors.YELLOW}Ready to experience AI that never forgets? Start a new session!{Colors.END}
""")

def main():
    """Main onboarding flow"""
    try:
        print_header()
        user_info = get_user_input()
        
        success = run_setup_steps(user_info)
        
        if success:
            print_success()
            return 0
        else:
            print(f"\n{Colors.RED}Setup incomplete. Please check the errors above.{Colors.END}")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Setup cancelled. Run 'context-engine setup' again anytime.{Colors.END}")
        return 1
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.END}")
        return 1

if __name__ == "__main__":
    sys.exit(main())