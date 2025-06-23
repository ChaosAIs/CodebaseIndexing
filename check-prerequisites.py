#!/usr/bin/env python3
"""Check if all prerequisites are installed for the Codebase Indexing Solution."""

import subprocess
import sys
import shutil

def run_command(command):
    """Run a command and return the output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def check_python():
    """Check Python version."""
    print("🐍 Checking Python...")
    success, output, _ = run_command("python3 --version")
    if success:
        version_str = output.split()[1]
        major, minor = map(int, version_str.split('.')[:2])
        if major >= 3 and minor >= 8:
            print(f"   ✅ Python {version_str} (OK)")
            return True
        else:
            print(f"   ❌ Python {version_str} (Need 3.8+)")
            return False
    else:
        print("   ❌ Python 3 not found")
        return False

def check_node():
    """Check Node.js version."""
    print("📦 Checking Node.js...")
    success, output, _ = run_command("node --version")
    if success:
        version_str = output.replace('v', '')
        major = int(version_str.split('.')[0])
        if major >= 16:
            print(f"   ✅ Node.js {version_str} (OK)")
            return True
        else:
            print(f"   ❌ Node.js {version_str} (Need 16+)")
            return False
    else:
        print("   ❌ Node.js not found")
        return False

def check_docker():
    """Check Docker."""
    print("🐳 Checking Docker...")
    if shutil.which("docker"):
        success, output, _ = run_command("docker --version")
        if success:
            print(f"   ✅ {output}")

            # Check if Docker is running
            success, _, _ = run_command("docker info")
            if success:
                print("   ✅ Docker daemon is running")
                return True
            else:
                print("   ❌ Docker daemon is not running")
                return False
        else:
            print("   ❌ Docker command failed")
            return False
    else:
        print("   ❌ Docker not found")
        return False

def check_docker_compose():
    """Check Docker Compose."""
    print("🔧 Checking Docker Compose...")
    if shutil.which("docker-compose"):
        success, output, _ = run_command("docker-compose --version")
        if success:
            print(f"   ✅ {output}")
            return True
        else:
            print("   ❌ Docker Compose command failed")
            return False
    else:
        print("   ❌ Docker Compose not found")
        return False

def check_git():
    """Check Git."""
    print("📝 Checking Git...")
    if shutil.which("git"):
        success, output, _ = run_command("git --version")
        if success:
            print(f"   ✅ {output}")
            return True
        else:
            print("   ❌ Git command failed")
            return False
    else:
        print("   ❌ Git not found")
        return False

def main():
    """Main function to check all prerequisites."""
    print("🔍 Checking Prerequisites for Codebase Indexing Solution\n")
    
    checks = [
        check_python(),
        check_node(),
        check_docker(),
        check_docker_compose(),
        check_git()
    ]
    
    print("\n" + "="*50)
    
    if all(checks):
        print("🎉 All prerequisites are installed and ready!")
        print("\nNext steps:")
        print("1. Run: ./scripts/setup.sh")
        print("2. Configure: cp .env.example .env && edit .env")
        print("3. Start: ./scripts/start.sh")
        return True
    else:
        print("❌ Some prerequisites are missing or outdated.")
        print("\nInstallation guides:")
        print("- Python 3.8+: https://www.python.org/downloads/")
        print("- Node.js 16+: https://nodejs.org/")
        print("- Docker: https://docs.docker.com/get-docker/")
        print("- Git: https://git-scm.com/downloads")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
