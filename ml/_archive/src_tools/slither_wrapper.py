from __future__ import annotations

"""
Slither Wrapper for Smart Contract Analysis

Production-grade wrapper with:
- Smart Solidity version detection
- Auto-installation of compiler versions
- Safe parallel processing (version-grouped)
- Comprehensive error handling
- Detailed logging for debugging

Author: Ali Motafegh
Project: SENTINEL
"""

import re
import logging
import subprocess
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Impact(Enum):
    """Vulnerability impact levels."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFORMATIONAL = "Informational"
    OPTIMIZATION = "Optimization"


class Confidence(Enum):
    """Detection confidence levels."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class ExecutionContext(Enum):
    """Execution environment for Slither analysis."""
    DOCKER = "docker"
    LOCAL = "local"


@dataclass
class Finding:
    """Single vulnerability detected by Slither."""
    check: str
    impact: Impact
    confidence: Confidence
    description: str
    lines: List[int] = field(default_factory=list)


@dataclass
class SlitherResult:
    """Result of running Slither on one contract."""
    contract_path: str
    success: bool
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    findings: List[Finding] = field(default_factory=list)
    analysis_time: float = 0.0
    detected_version: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def high_impact_count(self) -> int:
        """Count high-impact vulnerabilities."""
        return sum(1 for f in self.findings if f.impact == Impact.HIGH)
    
    def medium_impact_count(self) -> int:
        """Count medium-impact vulnerabilities."""
        return sum(1 for f in self.findings if f.impact == Impact.MEDIUM)
    
    def vulnerability_types(self) -> List[str]:
        """Get unique vulnerability types found."""
        return list(set(f.check for f in self.findings))


class SlitherWrapper:
    """Wrapper for Slither static analyzer with smart version management.
    
    Features:
    - Auto-detects Solidity version from pragma
    - Auto-installs missing compiler versions via solc-select
    - Safe parallel processing (groups by version to avoid race conditions)
    - Comprehensive error handling and logging
    - Batch processing with progress bars
    
    Args:
        timeout: Maximum seconds per contract (default: 30)
        min_impact: Minimum impact to include (default: "Medium")
        verbose: Enable detailed logging (default: True)
        auto_install: Auto-install missing solc versions (default: True)
    """
    
    def __init__(
        self,
        timeout: int = 30,
        min_impact: str = "Medium",
        verbose: bool = True,
        auto_install: bool = True,
        use_docker: bool = True  # NEW: Enable Docker by default
    ):
        self.timeout = timeout
        self.min_impact = min_impact.upper()
        self.verbose = verbose
        self.auto_install = auto_install
        
        # NEW: Docker support
        self.use_docker = use_docker
        self._docker_available = None  # Lazy initialization
        self._docker_container_name = None  # Track running container
        
        # Cache for installed versions (avoid repeated checks)
        self._installed_versions_cache = None
        
        if self.verbose:
            print(f"🔧 SlitherWrapper initialized:")
            print(f"   - Timeout: {timeout}s")
            print(f"   - Min impact: {min_impact}")
            print(f"   - Auto-install: {auto_install}")
            print(f"   - Use Docker: {use_docker}")  # NEW
            print()
# ============================================================================
# DOCKER INFRASTRUCTURE
# ============================================================================

    def _check_docker_available(self) -> bool:
        """Check if Docker is available and running.
        
        Checks:
        1. Docker command exists
        2. Docker daemon is running
        3. Can list containers
        
        Returns:
            True if Docker is fully available, False otherwise
        """
        if self._docker_available is not None:
            return self._docker_available
        
        try:
            # Check docker command exists
            if not shutil.which('docker'):
                if self.verbose:
                    print("⚠️  Docker command not found")
                self._docker_available = False
                return False
            
            # Check Docker daemon is running
            result = subprocess.run(
                ['docker', 'ps'],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode != 0:
                if self.verbose:
                    print("⚠️  Docker daemon not running")
                self._docker_available = False
                return False
            
            # Check if our image exists
            result = subprocess.run(
                ['docker', 'images', '-q', 'slither-analyzer:latest'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if not result.stdout.strip():
                if self.verbose:
                    print("⚠️  Docker image 'slither-analyzer:latest' not found")
                    print("   Build it with: docker build -t slither-analyzer:latest -f ml/docker/Dockerfile.slither ml/docker/")
                self._docker_available = False
                return False
            
            self._docker_available = True
            return True
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            if self.verbose:
                print(f"⚠️  Docker check failed: {e}")
            self._docker_available = False
            return False

    def _get_project_root(self) -> Path:
        """Get project root directory (sentinel/).
        
        Returns:
            Path to sentinel/ directory
        """
        # This file is at: sentinel/ml/src/tools/slither_wrapper.py
        return Path(__file__).parent.parent.parent.parent
    



    def _get_slither_command(self) -> str:
        """Get full path to slither command.
        
        Uses sys.executable to find the virtual environment and locate slither.
        
        Returns:
            Full path to slither executable
        """
        import sys
        
        # Method 1: Try shutil.which in current process
        slither_path = shutil.which('slither')
        if slither_path and Path(slither_path).exists():
            return slither_path
        
        # Method 2: Construct path from sys.executable (virtual environment)
        # sys.executable is /path/to/venv/bin/python
        # slither should be at /path/to/venv/bin/slither
        venv_bin = Path(sys.executable).parent
        slither_venv = venv_bin / 'slither'
        
        if slither_venv.exists():
            return str(slither_venv)
        
        # Method 3: Try common locations
        possible_paths = [
            Path.home() / '.local' / 'bin' / 'slither',
            Path('/usr/local/bin/slither'),
            Path('/usr/bin/slither')
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        # Fallback: return 'slither' and let it fail with clear error
        logger.warning("Could not find slither executable, using 'slither' (may fail in parallel execution)")
        return 'slither'



    def _start_docker_container(self) -> bool:
        """Start long-running Docker container for Slither analysis.
        
        Returns:
            True if container started successfully, False otherwise
        """
        if self._docker_container_name is not None:
            # Container already running
            return True
        
        try:
            project_root = self._get_project_root()
            ml_dir = project_root / "ml"
            
            if not ml_dir.exists():
                if self.verbose:
                    print(f"❌ ML directory not found: {ml_dir}")
                return False
            
            # Generate unique container name
            container_name = f"slither-analyzer-{os.getpid()}"
            
            if self.verbose:
                print(f"🐳 Starting Docker container: {container_name}")
                print(f"   Volume mount: {ml_dir} → /workspace")
            
            # Start container
            cmd = [
                'docker', 'run',
                '-d',  # Detached
                '--name', container_name,
                '-v', f'{ml_dir}:/workspace',  # Volume mount
                'slither-analyzer:latest',
                'tail', '-f', '/dev/null'  # Keep alive
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                if self.verbose:
                    print(f"❌ Failed to start container: {result.stderr}")
                return False
            
            self._docker_container_name = container_name
            
            if self.verbose:
                print(f"✅ Container started: {container_name}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Container startup failed: {e}")
            return False

    def _stop_docker_container(self):
        """Stop and remove Docker container."""
        if self._docker_container_name is None:
            return
        
        try:
            if self.verbose:
                print(f"🧹 Stopping Docker container: {self._docker_container_name}")
            
            # Stop container
            subprocess.run(
                ['docker', 'stop', self._docker_container_name],
                capture_output=True,
                timeout=10
            )
            
            # Remove container
            subprocess.run(
                ['docker', 'rm', self._docker_container_name],
                capture_output=True,
                timeout=5
            )
            
            self._docker_container_name = None
            
            if self.verbose:
                print("✅ Container stopped and removed")
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Container cleanup warning: {e}")

    def _translate_path_to_container(self, host_path: str) -> str:
        """Translate host path to container path.
        
        Example:
            /home/user/sentinel/ml/data/contracts/test.sol
            → /workspace/data/contracts/test.sol
        
        Args:
            host_path: Path on host machine
            
        Returns:
            Equivalent path inside container
        """
        host_path = Path(host_path).resolve()
        ml_dir = self._get_project_root() / "ml"
        
        try:
            # Get relative path from ml/ directory
            relative = host_path.relative_to(ml_dir)
            # Container path
            return f"/workspace/{relative}"
        except ValueError:
            # Path is not under ml/ - return as-is and let it fail
            return str(host_path)

    def _switch_version_docker(self, version: str) -> bool:
        """Switch Solidity version inside Docker container.
        
        If version not installed, attempts to install it first.
        
        Args:
            version: Solidity version (e.g., "0.8.20")
            
        Returns:
            True if successful, False otherwise
        """
        if not self._docker_container_name:
            return False
        
        try:
            # Try to switch to version
            cmd = [
                'docker', 'exec',
                self._docker_container_name,
                'solc-select', 'use', version
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                if self.verbose:
                    print(f"✅ Docker: Switched to Solidity {version}")
                return True
            
            # Switch failed - try to install first
            if "not installed" in result.stderr.lower() or "invalid" in result.stderr.lower():
                if self.verbose:
                    print(f"📥 Docker: Installing Solidity {version}...")
                
                install_cmd = [
                    'docker', 'exec',
                    self._docker_container_name,
                    'solc-select', 'install', version
                ]
                
                install_result = subprocess.run(
                    install_cmd,
                    capture_output=True,
                    text=True,
                    timeout=180  # Installation can take time
                )
                
                if install_result.returncode == 0:
                    # Installation succeeded, try switch again
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        if self.verbose:
                            print(f"✅ Docker: Installed and switched to Solidity {version}")
                        return True
            
            if self.verbose:
                print(f"⚠️  Docker: Failed to switch to {version}: {result.stderr[:100]}")
            return False
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Docker version switch error: {e}")
            return False



    def _execute_command(
        self,
        cmd: List[str],
        context: ExecutionContext,
        timeout: int = None
    ) -> subprocess.CompletedProcess:
        """Execute command in specified context (Docker or Local).
        
        Args:
            cmd: Command to execute
            context: ExecutionContext.DOCKER or ExecutionContext.LOCAL
            timeout: Command timeout (uses self.timeout if None)
            
        Returns:
            subprocess.CompletedProcess result
        """
        if timeout is None:
            timeout = self.timeout
        
        if context == ExecutionContext.DOCKER:
            # Wrap command in docker exec
            docker_cmd = [
                'docker', 'exec',
                self._docker_container_name
            ] + cmd
            
            return subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        else:
            # Execute locally with current environment
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy()  # Explicitly pass environment
            )


    def _build_slither_command_unified(
        self,
        contract_path: str,
        output_json: Path,
        solc_version: Optional[str],
        context: ExecutionContext
    ) -> List[str]:
        """Build Slither command for specified execution context.
        
        Args:
            contract_path: Path to contract (host path for LOCAL, already translated for DOCKER)
            output_json: Output JSON path (host path for LOCAL, container path for DOCKER)
            solc_version: Detected Solidity version
            context: Execution context
            
        Returns:
            Command as list of strings
        """
        if context == ExecutionContext.DOCKER:
            # Docker: Use simple 'slither' (it's in container PATH)
            cmd = ['slither', contract_path, '--json', str(output_json)]
        else:
            # Local: Use full path to slither
            slither_cmd = self._get_slither_command()
            cmd = [slither_cmd, str(contract_path), '--json', str(output_json)]
        
        # Add version-specific flags
        if solc_version:
            major_minor = '.'.join(solc_version.split('.')[:2])
            
            # For old versions, disable some checks that cause issues
            if major_minor in ['0.4', '0.5']:
                cmd.extend([
                    '--solc-disable-warnings',
                    '--filter-paths', 'node_modules'
                ])
        
        return cmd


    def _determine_execution_context(self) -> ExecutionContext:
        """Determine which execution context to use.
        
        Logic:
        1. If use_docker=False → LOCAL
        2. If use_docker=True and Docker available → DOCKER
        3. If use_docker=True but Docker unavailable → LOCAL (fallback)
        
        Returns:
            ExecutionContext enum value
        """
        if not self.use_docker:
            return ExecutionContext.LOCAL
        
        if self._check_docker_available():
            return ExecutionContext.DOCKER
        else:
            if self.verbose:
                print("⚠️  Docker requested but not available, using local execution")
            return ExecutionContext.LOCAL

    # ============================================================================
    # UNIFIED EXECUTION ENGINE
    # ============================================================================

    def _run_slither_unified(
        self,
        contract_path: str,
        detected_version: Optional[str],
        context: ExecutionContext
    ) -> SlitherResult:
        """Run Slither analysis in specified execution context.
        
        This is the unified execution method that handles both Docker and Local.
        
        Args:
            contract_path: Path to contract (host path)
            detected_version: Detected Solidity version
            context: ExecutionContext.DOCKER or ExecutionContext.LOCAL
            
        Returns:
            SlitherResult object
        """
        contract_path = Path(contract_path)
        start_time = time.time()
        
        try:
            # STEP 1: Ensure compiler version is available
            if detected_version:
                if context == ExecutionContext.DOCKER:
                    # Docker: Switch version inside container
                    if not self._switch_version_docker(detected_version):
                        if self.verbose:
                            print(f"⚠️  Version switch failed, trying anyway...")
                else:
                    # Local: Use existing version management
                    if self.auto_install:
                        self._ensure_solc_version(detected_version)
            
            # STEP 2: Prepare paths based on context
            if context == ExecutionContext.DOCKER:
                # Translate paths to container paths
                container_contract_path = self._translate_path_to_container(str(contract_path))
                output_json_name = f"{contract_path.stem}_slither_{int(time.time())}.json"
                output_json_host = contract_path.parent / output_json_name
                output_json_container = self._translate_path_to_container(str(output_json_host))
                
                slither_contract_path = container_contract_path
                slither_output_json = output_json_container
                output_json = output_json_host  # For reading results
            else:
                # Local: Use host paths
                output_json = contract_path.parent / f"{contract_path.stem}_slither_{int(time.time())}.json"
                slither_contract_path = str(contract_path)
                slither_output_json = str(output_json)
            
            # STEP 3: Build command
            cmd = self._build_slither_command_unified(
                slither_contract_path,
                Path(slither_output_json),
                detected_version,
                context
            )
            
            if self.verbose:
                if context == ExecutionContext.DOCKER:
                    print(f"🐳 Running Slither in Docker container...")
                else:
                    print(f"💻 Running Slither locally...")
                print(f"   Command: {' '.join(cmd[:3])}...")
            
            # STEP 4: Execute command
            result = self._execute_command(cmd, context, timeout=self.timeout)
            
            analysis_time = time.time() - start_time
            
            if self.verbose:
                print(f"⏱️  Analysis completed in {analysis_time:.2f}s")
            
            # STEP 5: Parse results
            if output_json.exists():
                if self.verbose:
                    print(f"✅ JSON output found, parsing results...")
                
                with open(output_json, 'r') as f:
                    json_data = json.load(f)
                
                slither_result = self._parse_output(json_data, str(contract_path))
                slither_result.analysis_time = analysis_time
                slither_result.detected_version = detected_version
                
                # Clean up temp file
                output_json.unlink()
                
                if self.verbose:
                    print(f"✅ Success! Found {len(slither_result.findings)} findings")
                
                return slither_result
            else:
                # No JSON output - compilation likely failed
                if self.verbose:
                    print(f"❌ Compilation failed")
                    if result.stderr:
                        print(f"📄 Error preview: {result.stderr[:200]}...")
                
                # If Docker failed, try local fallback
                if context == ExecutionContext.DOCKER:
                    if self.verbose:
                        print(f"🔄 Docker execution failed, falling back to local...")
                    return self._run_slither_unified(
                        str(contract_path),
                        detected_version,
                        ExecutionContext.LOCAL
                    )
                
                return SlitherResult(
                    contract_path=str(contract_path),
                    success=False,
                    error_type="compilation_error",
                    error_message=result.stderr[:500] if result.stderr else "Unknown compilation error",
                    analysis_time=analysis_time,
                    detected_version=detected_version
                )
        
        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"⏰ Timeout after {self.timeout}s")
            
            # If Docker timed out, try local fallback
            if context == ExecutionContext.DOCKER:
                if self.verbose:
                    print(f"🔄 Docker timeout, falling back to local...")
                return self._run_slither_unified(
                    str(contract_path),
                    detected_version,
                    ExecutionContext.LOCAL
                )
            
            return SlitherResult(
                contract_path=str(contract_path),
                success=False,
                error_type="timeout",
                error_message=f"Analysis timeout after {self.timeout}s",
                detected_version=detected_version
            )
        
        except Exception as e:
            if self.verbose:
                print(f"💥 Unexpected error: {type(e).__name__}: {e}")
            
            # If Docker crashed, try local fallback
            if context == ExecutionContext.DOCKER:
                if self.verbose:
                    print(f"🔄 Docker error, falling back to local...")
                return self._run_slither_unified(
                    str(contract_path),
                    detected_version,
                    ExecutionContext.LOCAL
                )
            
            return SlitherResult(
                contract_path=str(contract_path),
                success=False,
                error_type="execution_error",
                error_message=str(e),
                detected_version=detected_version
            )

    
    def run(self, contract_path: str) -> SlitherResult:
        """Run Slither on a single contract with smart version management.
        
        Automatically uses Docker if available, falls back to local execution.
        
        Args:
            contract_path: Path to .sol file
            
        Returns:
            SlitherResult with findings or error
        """
        contract_path = Path(contract_path)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Analyzing: {contract_path.name}")
            print(f"{'='*80}")
        
        # Validate file exists
        if not contract_path.exists():
            if self.verbose:
                print(f"❌ File not found: {contract_path}")
            return SlitherResult(
                contract_path=str(contract_path),
                success=False,
                error_type="not_found",
                error_message=f"Contract not found: {contract_path}"
            )
        
        # Step 1: Detect Solidity version
        detected_version = self._detect_solidity_version(contract_path)
        if self.verbose:
            if detected_version:
                print(f"🔍 Detected Solidity version: {detected_version}")
            else:
                print(f"⚠️  No pragma found, using default compiler")
        
        # Step 2: Determine execution context
        context = self._determine_execution_context()
        
        # Step 3: Start Docker container if needed
        if context == ExecutionContext.DOCKER:
            if not self._start_docker_container():
                if self.verbose:
                    print("⚠️  Failed to start Docker container, falling back to local")
                context = ExecutionContext.LOCAL
        
        # Step 4: Run Slither in determined context
        result = self._run_slither_unified(str(contract_path), detected_version, context)
        
        # Step 5: Cleanup (only for single-contract runs)
        if context == ExecutionContext.DOCKER:
            self._stop_docker_container()
        
        return result

    
    def _detect_solidity_version(self, contract_path: Path) -> Optional[str]:
        """Extract Solidity version from pragma statement.
        
        Enhanced to handle:
        - Different encodings (UTF-8, Latin-1, Windows-1252)
        - Hidden characters (BOM, tabs, multiple spaces)
        - Various pragma formats including ranges (>=0.4.22 <0.6.0)
        
        Args:
            contract_path: Path to contract file
            
        Returns:
            Version string (e.g., "0.4.24") or None
        """
        try:
            content = None
            # Try multiple encodings
            encodings = ['utf-8', 'latin-1', 'windows-1252', 'utf-8-sig']
            
            for encoding in encodings:
                try:
                    with open(contract_path, 'r', encoding=encoding) as f:
                        content = f.read(5000)  # Read more chars
                    break  # Success
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if not content:
                if self.verbose:
                    print(f"⚠️  Could not read file with any encoding")
                return None
            
            # Normalize whitespace (replace tabs, multiple spaces)
            content = re.sub(r'\s+', ' ', content)
            
            # Pattern 1: ^0.4.24 or ~0.4.24
            # Pattern 1: ^0.4.24 (caret - means range)
            pattern1 = r'pragma\s+solidity\s*\^\s*(\d+)\.(\d+)\.(\d+)'
            match = re.search(pattern1, content, re.IGNORECASE)
            if match:
                major = match.group(1)
                minor = match.group(2)
                # For caret, use highest version in that major.minor series
                major_minor = f"{major}.{minor}"
                
                # Prefer highest available in this series
                version_map = {
                    "0.4": "0.4.26",
                    "0.5": "0.5.17",
                    "0.6": "0.6.12",
                    "0.7": "0.7.6",
                    "0.8": "0.8.20"
                }
                
                if major_minor in version_map:
                    return version_map[major_minor]
                else:
                    # Fallback to patch version specified
                    return f"{major}.{minor}.{match.group(3)}"

            # Pattern 1b: ~0.4.24 (tilde - use exact version)
            pattern1b = r'pragma\s+solidity\s*~\s*(\d+\.\d+\.\d+)'
            match = re.search(pattern1b, content, re.IGNORECASE)
            if match:
                return match.group(1)

            
            # Pattern 2: >=0.4.22 <0.6.0 (range - extract BOTH bounds)
            pattern2 = r'pragma\s+solidity\s*>=\s*(\d+\.\d+\.\d+)\s*<\s*(\d+\.\d+)'
            match = re.search(pattern2, content, re.IGNORECASE)
            if match:
                lower = match.group(1)  # e.g., "0.4.22"
                upper = match.group(2)  # e.g., "0.6"
                
                # Use highest version in lower bound's series that's below upper
                lower_parts = lower.split('.')
                upper_parts = upper.split('.')
                
                lower_major = int(lower_parts[0])
                lower_minor = int(lower_parts[1])
                upper_major = int(upper_parts[0])
                upper_minor = int(upper_parts[1]) if len(upper_parts) > 1 else 0
                
                # If range spans multiple minor versions, prefer one before upper bound
                if lower_major == upper_major and upper_minor > lower_minor:
                    # Use (upper_minor - 1).17 as a safe bet
                    target_minor = upper_minor - 1
                    return f"{lower_major}.{target_minor}.17"
                else:
                    return lower
            
            # Pattern 3: Exact version 0.5.16
            pattern3 = r'pragma\s+solidity\s+(\d+\.\d+\.\d+)'
            match = re.search(pattern3, content, re.IGNORECASE)
            if match:
                return match.group(1)
            
            # Pattern 4: Two-part version (0.4, 0.5) - add .0
            pattern4 = r'pragma\s+solidity\s*[\^~]?\s*(\d+\.\d+)\s*[;\s]'
            match = re.search(pattern4, content, re.IGNORECASE)
            if match:
                version = match.group(1)
                return f"{version}.0"
            
            # Pattern 5: Very permissive - catch any version-like string
            pattern5 = r'pragma\s+solidity\s*[\^~>=<\s]*(\d+\.\d+(?:\.\d+)?)'
            match = re.search(pattern5, content, re.IGNORECASE)
            if match:
                version = match.group(1)
                # Ensure 3-part version
                parts = version.split('.')
                if len(parts) == 2:
                    return f"{version}.0"
                return version
            
            return None
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error reading file: {e}")
            return None
    def _get_highest_compatible_version(self, major_minor: str) -> Optional[str]:
        """Get highest available version in major.minor series.
        
        Example: major_minor="0.4" → returns "0.4.26" if available
        """
        # Known versions in Docker (ordered high to low)
        available_versions = [
            "0.8.20", "0.8.0", "0.7.6", "0.6.12",
            "0.5.17", "0.5.8", "0.5.7", "0.5.4", "0.5.2", "0.5.1", "0.5.0",
            "0.4.26", "0.4.25", "0.4.24", "0.4.23", "0.4.21", "0.4.20",
            "0.4.19", "0.4.18", "0.4.17", "0.4.16", "0.4.15", "0.4.12",
            "0.4.11", "0.4.8", "0.4.4", "0.4.2"
        ]
        
        # Find highest version matching major.minor
        for version in available_versions:
            if version.startswith(major_minor + "."):
                if self.verbose:
                    print(f"   📌 ^{major_minor}.X → using {version}")
                return version
        
        # Fallback: return major_minor + ".0"
        return f"{major_minor}.0"

    def _ensure_solc_version(self, version: str) -> bool:
        """Ensure specific Solidity compiler version is installed.
        
        Strategy:
        1. Check if exact version installed → use it
        2. Try to install exact version
        3. If install fails, find nearest compatible version
        4. Compatible = same major.minor, closest patch (±2 patches)
        
        Args:
            version: Solidity version (e.g., "0.4.24")
            
        Returns:
            True if version is available, False otherwise
        """
        if not version:
            return False
        
        try:
            # Check if solc-select is installed
            check_result = subprocess.run(
                ['solc-select', 'versions'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if check_result.returncode != 0:
                if self.verbose:
                    print(f"⚠️  solc-select not installed. Install with: pip install solc-select")
                return False
            
            # Get list of installed versions (use cache)
            if self._installed_versions_cache is None:
                self._installed_versions_cache = check_result.stdout
            
            installed_versions = self._installed_versions_cache
            
            # STEP 1: Check if exact version already installed
            if version in installed_versions:
                if self.verbose:
                    print(f"✅ Solidity {version} already installed")
                subprocess.run(
                    ['solc-select', 'use', version],
                    capture_output=True,
                    timeout=5
                )
                return True
            
            # STEP 2: Try to install exact version
            if self.verbose:
                print(f"📥 Installing Solidity {version}... (this may take 30-60s)")
            
            install_result = subprocess.run(
                ['solc-select', 'install', version],
                capture_output=True,
                text=True,
                timeout=180
            )
            
            if install_result.returncode == 0:
                # Installation succeeded!
                self._installed_versions_cache = None  # Clear cache
                subprocess.run(
                    ['solc-select', 'use', version],
                    capture_output=True,
                    timeout=5
                )
                if self.verbose:
                    print(f"✅ Successfully installed and activated Solidity {version}")
                return True
            
            # STEP 3: Exact install failed - try compatible version
            if self.verbose:
                print(f"⚠️  Failed to install {version}, looking for compatible version...")
                print(f"   Error: {install_result.stderr[:150]}")
            
            compatible = self._get_smart_compatible_version(version, installed_versions)
            if compatible:
                if self.verbose:
                    print(f"✅ Using compatible version {compatible} (closest to {version})")
                subprocess.run(
                    ['solc-select', 'use', compatible],
                    capture_output=True,
                    timeout=5
                )
                return True
            
            # STEP 4: No compatible version - try to install nearest available
            nearest = self._get_nearest_installable_version(version)
            if nearest and nearest != version:
                if self.verbose:
                    print(f"📥 Installing nearest available version {nearest}...")
                
                install_result = subprocess.run(
                    ['solc-select', 'install', nearest],
                    capture_output=True,
                    text=True,
                    timeout=180
                )
                
                if install_result.returncode == 0:
                    self._installed_versions_cache = None
                    subprocess.run(
                        ['solc-select', 'use', nearest],
                        capture_output=True,
                        timeout=5
                    )
                    if self.verbose:
                        print(f"✅ Installed {nearest} as alternative to {version}")
                    return True
            
            if self.verbose:
                print(f"❌ Could not find or install compatible version for {version}")
            return False
            
        except FileNotFoundError:
            if self.verbose:
                print(f"⚠️  solc-select not found. Install with: pip install solc-select")
            return False
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Version management error: {e}")
            return False
    
    def _get_smart_compatible_version(
        self,
        requested_version: str,
        installed_versions: str
    ) -> Optional[str]:
        """Find smartly compatible installed version.
        
        Compatible definition:
        - Same major.minor (e.g., 0.4.x for 0.4.21)
        - Within ±2 patch versions (0.4.19-0.4.23 for 0.4.21)
        - Prefer higher patch (0.4.22 over 0.4.20)
        
        Args:
            requested_version: Requested version (e.g., "0.4.21")
            installed_versions: Output from solc-select versions
            
        Returns:
            Best compatible version or None
        """
        try:
            # Parse installed versions
            installed = []
            for line in installed_versions.split('\n'):
                parts = line.strip().split()
                if parts and parts[0].count('.') == 2:
                    installed.append(parts[0])
            
            if not installed:
                return None
            
            # Parse requested version
            req_parts = requested_version.split('.')
            req_major = int(req_parts[0])
            req_minor = int(req_parts[1])
            req_patch = int(req_parts[2])
            
            # Find compatible versions (same major.minor, within ±2 patches)
            compatible = []
            for ver in installed:
                ver_parts = ver.split('.')
                ver_major = int(ver_parts[0])
                ver_minor = int(ver_parts[1])
                ver_patch = int(ver_parts[2])
                
                # Must match major.minor
                if ver_major != req_major or ver_minor != req_minor:
                    continue
                
                # Within ±2 patch versions
                patch_diff = abs(ver_patch - req_patch)
                if patch_diff <= 2:
                    compatible.append((ver, patch_diff, ver_patch))
            
            if not compatible:
                return None
            
            # Sort by: 1) smallest patch diff, 2) prefer higher patch
            compatible.sort(key=lambda x: (x[1], -x[2]))
            return compatible[0][0]
            
        except Exception:
            return None
    
    def _get_nearest_installable_version(self, requested_version: str) -> Optional[str]:
        """Find nearest installable version from solc-select.
        
        Args:
            requested_version: Requested version (e.g., "0.4.21")
            
        Returns:
            Nearest installable version or None
        """
        try:
            req_parts = requested_version.split('.')
            major = req_parts[0]
            minor = req_parts[1]
            patch = int(req_parts[2])
            
            # Try versions: requested+1, requested-1, requested+2, requested-2
            candidates = [
                f"{major}.{minor}.{patch + 1}",
                f"{major}.{minor}.{patch - 1}" if patch > 0 else None,
                f"{major}.{minor}.{patch + 2}",
                f"{major}.{minor}.{patch - 2}" if patch > 1 else None,
            ]
            
            # Return first non-None candidate
            for candidate in candidates:
                if candidate:
                    return candidate
            
            return None
            
        except Exception:
            return None
    
    def _build_slither_command(
        self,
        contract_path: Path,
        output_json: Path,
        solc_version: Optional[str]
    ) -> List[str]:
        """Build Slither command with appropriate arguments.
        
        Args:
            contract_path: Path to contract
            output_json: Output JSON path
            solc_version: Detected Solidity version
            
        Returns:
            Command as list of strings
        """
        cmd = ['slither', str(contract_path), '--json', str(output_json)]
        
        if solc_version:
            major_minor = '.'.join(solc_version.split('.')[:2])
            
            # For old versions, disable some checks that cause issues
            if major_minor in ['0.4', '0.5']:
                cmd.extend([
                    '--solc-disable-warnings',
                    '--filter-paths', 'node_modules'
                ])
        
        return cmd
    
    def _run_fallback(
        self,
        contract_path: Path,
        previous_time: float,
        stderr: str
    ) -> SlitherResult:
        """Fallback: try running without version-specific args.
        
        Args:
            contract_path: Path to contract
            previous_time: Time spent on first attempt
            stderr: Error from first attempt
            
        Returns:
            SlitherResult from fallback attempt
        """
        output_json = contract_path.parent / f"{contract_path.stem}_slither_fallback.json"
        cmd = ['slither', str(contract_path), '--json', str(output_json)]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if output_json.exists():
                with open(output_json, 'r') as f:
                    json_data = json.load(f)
                
                slither_result = self._parse_output(json_data, str(contract_path))
                slither_result.analysis_time = previous_time
                output_json.unlink()
                
                if self.verbose:
                    print(f"✅ Fallback succeeded!")
                return slither_result
        except Exception:
            pass
        
        # Both attempts failed
        return SlitherResult(
            contract_path=str(contract_path),
            success=False,
            error_type="compilation_error",
            error_message=stderr[:500] if stderr else "Compilation failed (tried fallback)",
            analysis_time=previous_time
        )
    
    def _parse_output(self, json_data: dict, contract_path: str) -> SlitherResult:
        """Parse Slither JSON output into SlitherResult.
        
        Args:
            json_data: Parsed JSON from Slither
            contract_path: Path to analyzed contract
            
        Returns:
            SlitherResult with parsed findings
        """
        findings = []
        
        if 'results' not in json_data or 'detectors' not in json_data['results']:
            return SlitherResult(
                contract_path=contract_path,
                success=True,
                findings=[]
            )
        
        for detector in json_data['results']['detectors']:
            try:
                impact_str = detector.get('impact', 'Informational')
                confidence_str = detector.get('confidence', 'Low')
                
                # Filter by minimum impact
                if not self._meets_min_impact(impact_str):
                    continue
                
                # Extract line numbers
                lines = self._extract_lines(detector.get('elements', []))
                
                finding = Finding(
                    check=detector.get('check', 'unknown'),
                    impact=Impact(impact_str),
                    confidence=Confidence(confidence_str),
                    description=detector.get('description', ''),
                    lines=lines
                )
                findings.append(finding)
                
            except (KeyError, ValueError):
                continue
        
        return SlitherResult(
            contract_path=contract_path,
            success=True,
            findings=findings
        )
    
    def _meets_min_impact(self, impact_str: str) -> bool:
        """Check if impact meets minimum threshold."""
        impact_order = {
            'INFORMATIONAL': 0,
            'OPTIMIZATION': 0,
            'LOW': 1,
            'MEDIUM': 2,
            'HIGH': 3
        }
        
        current = impact_order.get(impact_str.upper(), 0)
        minimum = impact_order.get(self.min_impact, 2)
        return current >= minimum
    
    def _extract_lines(self, elements: List[dict]) -> List[int]:
        """Extract line numbers from detector elements."""
        lines = set()
        for element in elements:
            if 'source_mapping' in element and 'lines' in element['source_mapping']:
                lines.update(element['source_mapping']['lines'])
        return sorted(list(lines))
    
    def run_batch(
    self,
    contract_paths: List[str],
    save_dir: Optional[str] = None,
    max_workers: Optional[int] = None
    ) -> List[SlitherResult]:
        """Run Slither on multiple contracts with safe parallel processing.
        
        Groups contracts by Solidity version first to avoid race conditions
        with solc-select global state changes.
        
        For Docker execution: Starts container once at beginning, keeps it
        running for all analyses, cleans up at end.
        
        Args:
            contract_paths: Paths to contracts
            save_dir: Optional directory to save results
            max_workers: Number of parallel workers per version group
            
        Returns:
            List of SlitherResult objects
        """
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
        
        print("="*80)
        print("BATCH ANALYSIS (VERSION-GROUPED)")
        print("="*80)
        print(f"Contracts: {len(contract_paths)}")
        print(f"Parallel workers: {max_workers}")
        print(f"Timeout: {self.timeout}s per contract")
        print(f"Min impact: {self.min_impact}")
        print("="*80 + "\n")
        
        # Determine execution context
        context = self._determine_execution_context()
        
        # Start Docker container if needed (ONCE for entire batch)
        if context == ExecutionContext.DOCKER:
            if not self._start_docker_container():
                if self.verbose:
                    print("⚠️  Failed to start Docker container, falling back to local")
                context = ExecutionContext.LOCAL
            else:
                print(f"✅ Docker container ready for batch processing\n")
        
        try:
            # STEP 1: Group contracts by Solidity version
            print("📊 Phase 1: Detecting Solidity versions...")
            version_groups = {}
            for path in tqdm(contract_paths, desc="Version detection"):
                version = self._detect_solidity_version(Path(path))
                if version not in version_groups:
                    version_groups[version] = []
                version_groups[version].append(path)
            
            # Separate None (undetected) from valid versions
            valid_versions = {v: p for v, p in version_groups.items() if v is not None}
            undetected = version_groups.get(None, [])
            
            print(f"\n✅ Found {len(valid_versions)} different Solidity versions:")
            for version, paths in sorted(valid_versions.items()):
                print(f"   - v{version}: {len(paths)} contracts")
            if undetected:
                print(f"   - [No version detected]: {len(undetected)} contracts")
            
            # STEP 2: Process each version group sequentially
            all_results = []
            print(f"\n{'='*80}")
            print("🔧 Phase 2: Analyzing contracts by version group")
            print(f"{'='*80}\n")
            
            # Process valid versions
            for version, paths in sorted(valid_versions.items()):
                print(f"\n📦 Processing {len(paths)} contracts with Solidity v{version}...")
                
                # Switch version once for this group
                if context == ExecutionContext.DOCKER:
                    if not self._switch_version_docker(version):
                        # Version not available in Docker - fall back to local
                        print(f"⚠️  Version {version} not available in Docker")
                        print(f"🔄 Falling back to local execution for this group...")
                        
                        # Process this group with LOCAL context
                        original_verbose = self.verbose
                        self.verbose = False  # Reduce noise for parallel execution
                        
                        for path in tqdm(paths, desc=f"   v{version} (local)", leave=False):
                            # Ensure version locally
                            if self.auto_install:
                                self._ensure_solc_version(version)
                            
                            # Execute locally
                            result = self._run_slither_unified(path, version, ExecutionContext.LOCAL)
                            all_results.append(result)
                        
                        self.verbose = original_verbose
                        continue
                else:
                    # Local: ensure version is available
                    if self.auto_install and not self._ensure_solc_version(version):
                        logger.warning(f"Failed to install/switch to {version}, skipping group")
                        for path in paths:
                            all_results.append(SlitherResult(
                                contract_path=path,
                                success=False,
                                error_type="version_install_failed",
                                error_message=f"Failed to install Solidity {version}",
                                detected_version=version
                            ))
                        continue
                
                # Parallel process within this version group
                group_results = self._process_version_group(paths, version, max_workers, context)
                all_results.extend(group_results)
            
            # Handle undetected versions
            if undetected:
                print(f"\n⚠️  Skipping {len(undetected)} contracts with undetected versions")
                for path in undetected:
                    all_results.append(SlitherResult(
                        contract_path=path,
                        success=False,
                        error_type="no_version_detected",
                        error_message="Could not detect Solidity version from pragma"
                    ))
            
            # STEP 3: Generate summary and save
            self._print_summary(all_results)
            
            if save_dir:
                self._save_results(all_results, save_dir)
                self._save_failure_report(all_results, save_dir)
            
            return all_results
        
        finally:
            # ALWAYS cleanup Docker container (even if errors occur)
            if context == ExecutionContext.DOCKER:
                self._stop_docker_container()


    
    def _process_version_group(
        self,
        contract_paths: List[str],
        version: str,
        max_workers: int,
        context: ExecutionContext
    ) -> List[SlitherResult]:
        """Process contracts in parallel for a specific Solidity version.
        
        Safe to parallelize because all contracts use the same compiler version.
        
        Args:
            contract_paths: List of contract paths
            version: Solidity version for this group
            max_workers: Number of parallel workers
            context: Execution context (DOCKER or LOCAL)
            
        Returns:
            List of SlitherResult objects
        """
        results = []
        
        # Use ProcessPoolExecutor for CPU-bound work
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs for this version group
            future_to_path = {
                executor.submit(
                    self._analyze_single_no_switch,
                    path,
                    version,
                    context
                ): path
                for path in contract_paths
            }
            
            # Process as they complete
            with tqdm(total=len(contract_paths), desc=f"   v{version}", leave=False) as pbar:
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {path}: {e}")
                        results.append(SlitherResult(
                            contract_path=path,
                            success=False,
                            error_type="processing_error",
                            error_message=str(e),
                            detected_version=version
                        ))
                    pbar.update(1)
        
        return results

    
    def _analyze_single_no_switch(
        self,
        contract_path: str,
        version: str,
        context: ExecutionContext
    ) -> SlitherResult:
        """Analyze contract WITHOUT switching compiler version.
        
        Assumes correct version is already set globally.
        Used in parallel processing within version groups.
        
        Args:
            contract_path: Path to contract
            version: Solidity version (for metadata)
            context: Execution context
            
        Returns:
            SlitherResult object
        """
        # Disable verbose mode for parallel execution (avoid clutter)
        original_verbose = self.verbose
        self.verbose = False
        
        try:
            result = self._run_slither_unified(contract_path, version, context)
            return result
        finally:
            self.verbose = original_verbose

    
    def _print_summary(self, results: List[SlitherResult]):
        """Print comprehensive analysis summary."""
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        
        print(f"\n📊 Execution Statistics:")
        print(f"   Total contracts: {total}")
        print(f"   Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"   Failed: {failed} ({failed/total*100:.1f}%)")
        
        # Error breakdown
        if failed > 0:
            error_types = {}
            failed_contracts = []
            
            for r in results:
                if not r.success:
                    if r.error_type:
                        error_types[r.error_type] = error_types.get(r.error_type, 0) + 1
                    
                    failed_contracts.append({
                        'path': r.contract_path,
                        'error_type': r.error_type,
                        'error_message': r.error_message,
                        'detected_version': r.detected_version
                    })
            
            print(f"\n   ❌ Error breakdown:")
            for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
                print(f"      - {error_type}: {count}")
            
            # Detailed failure report (first 5)
            print(f"\n   📋 Sample failures (showing first 5):")
            print(f"   {'-'*76}")
            
            for i, failure in enumerate(failed_contracts[:5], 1):
                contract_name = Path(failure['path']).name
                print(f"\n   {i}. {contract_name}")
                print(f"      Detected version: {failure['detected_version'] or 'None'}")
                print(f"      Error type: {failure['error_type']}")
                
                error_msg = failure['error_message'] or 'No error message'
                if len(error_msg) > 150:
                    error_msg = error_msg[:150] + "..."
                print(f"      Error: {error_msg}")
        
        # Findings summary
        total_findings = sum(len(r.findings) for r in results if r.success)
        high_impact = sum(r.high_impact_count() for r in results if r.success)
        medium_impact = sum(r.medium_impact_count() for r in results if r.success)
        
        print(f"\n🔍 Findings:")
        print(f"   Total findings: {total_findings}")
        print(f"   High impact: {high_impact}")
        print(f"   Medium impact: {medium_impact}")
        
        # Vulnerability types
        all_types = {}
        for r in results:
            if r.success:
                for vuln_type in r.vulnerability_types():
                    all_types[vuln_type] = all_types.get(vuln_type, 0) + 1
        
        if all_types:
            print(f"\n   🐛 Top 10 vulnerability types:")
            for vuln_type, count in sorted(all_types.items(), key=lambda x: -x[1])[:10]:
                print(f"      - {vuln_type}: {count}")
        
        # Performance
        if successful > 0:
            avg_time = sum(r.analysis_time for r in results if r.success) / successful
            print(f"\n⏱️  Performance:")
            print(f"   Average analysis time: {avg_time:.2f}s per contract")
            print(f"   Total time: {sum(r.analysis_time for r in results):.1f}s")
        
        print("="*80)
    
    def _save_results(self, results: List[SlitherResult], save_dir: str):
        """Save results to JSON file."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        results_file = save_path / f'slither_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        # Convert results to dict
        results_data = []
        for r in results:
            results_data.append({
                'contract_path': r.contract_path,
                'success': r.success,
                'error_message': r.error_message,
                'error_type': r.error_type,
                'detected_version': r.detected_version,
                'findings': [
                    {
                        'check': f.check,
                        'impact': f.impact.value,
                        'confidence': f.confidence.value,
                        'description': f.description,
                        'lines': f.lines
                    }
                    for f in r.findings
                ],
                'analysis_time': r.analysis_time,
                'timestamp': r.timestamp
            })
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    def _save_failure_report(self, results: List[SlitherResult], save_dir: str):
        """Save detailed failure report to file."""
        failed_results = [r for r in results if not r.success]
        
        if not failed_results:
            return
        
        save_path = Path(save_dir)
        report_file = save_path / f'failure_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SLITHER ANALYSIS FAILURE REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total failures: {len(failed_results)}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            for i, result in enumerate(failed_results, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"FAILURE #{i}\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"Contract: {result.contract_path}\n")
                f.write(f"Detected version: {result.detected_version or 'None detected'}\n")
                f.write(f"Error type: {result.error_type}\n")
                f.write(f"Analysis time: {result.analysis_time:.2f}s\n\n")
                f.write(f"Full Error Message:\n")
                f.write(f"{'-'*80}\n")
                f.write(result.error_message or 'No error message')
                f.write(f"\n{'-'*80}\n\n")
        
        print(f"📄 Detailed failure report saved to: {report_file}")


# Example usage
if __name__ == '__main__':
    wrapper = SlitherWrapper(timeout=30, min_impact="High", verbose=True)
    
    # Single contract
    result = wrapper.run('path/to/contract.sol')
    print(f"\nFound {len(result.findings)} vulnerabilities")
    
    # Batch processing
    contracts = ['contract1.sol', 'contract2.sol', 'contract3.sol']
    results = wrapper.run_batch(contracts, save_dir='ml/data/slither_results/')
