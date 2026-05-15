"""
SlitherWrapperTurbo - Ultra-optimized version with parallel version processing

Optimizations:
- A: Reduced timeout (15s configurable)
- B: More workers (20 default)
- C: Skip ancient versions (0.3.x)
- D: RAM disk support (fast I/O)
- E: PARALLEL version group processing
"""

from ml.src.tools.slither_wrapper import SlitherWrapper, SlitherResult, ExecutionContext
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class SlitherWrapperTurbo(SlitherWrapper):
    """
    Turbocharged SlitherWrapper with parallel version processing.
    
    Key differences from base class:
    1. Process multiple Solidity versions in parallel (not sequential)
    2. Skip ancient/unsupported versions automatically
    3. Support RAM disk for faster temp file I/O
    """
    
    def __init__(
        self,
        timeout: int = 15,  # Optimization A: Reduced from 30s
        skip_ancient: bool = True,  # Optimization C
        temp_dir: str = "/tmp",  # Optimization D
        **kwargs
    ):
        super().__init__(timeout=timeout, **kwargs)
        self.skip_ancient = skip_ancient
        self.temp_dir = Path(temp_dir)
        
        # Verify temp dir (RAM disk)
        if not self.temp_dir.exists():
            logger.warning(f"Temp dir {temp_dir} not found, using /tmp")
            self.temp_dir = Path("/tmp")
    
    def _should_skip_version(self, version: Optional[str]) -> bool:
        """
        Optimization C: Skip ancient/unsupported versions
        
        Skip:
        - v0.3.x (not in solc-select)
        - None/unknown
        """
        if not self.skip_ancient:
            return False
        
        if not version:
            return True
        
        # Skip 0.3.x
        if version.startswith("0.3."):
            return True
        
        return False
    
    def run_batch(
        self,
        contract_paths: List[str],
        save_dir: Optional[str] = None,
        max_workers: int = 20  # Optimization B: Increased from default
    ) -> List[SlitherResult]:
        """
        TURBO BATCH PROCESSING with parallel version groups.
        
        Optimization E: Process multiple Solidity versions in PARALLEL
        instead of sequential.
        """
        print("="*80)
        print("⚡ TURBO MODE ACTIVATED")
        print("="*80)
        print(f"Contracts: {len(contract_paths)}")
        print(f"Workers per version: {max_workers}")
        print(f"Timeout: {self.timeout}s (Optimization A)")
        print(f"Skip ancient: {self.skip_ancient} (Optimization C)")
        print(f"Temp dir: {self.temp_dir} (Optimization D)")
        print(f"Parallel versions: ✅ (Optimization E)")
        print("="*80 + "\n")
        
        # Determine execution context
        context = self._determine_execution_context()
        
        # Start Docker container if needed
        if context == ExecutionContext.DOCKER:
            if not self._start_docker_container():
                if self.verbose:
                    print("⚠️ Docker failed, using local")
                context = ExecutionContext.LOCAL
            else:
                print("✅ Docker container ready\n")
        
        try:
            # PHASE 1: Version detection (super fast)
            print("📊 Phase 1: Detecting Solidity versions...")
            version_groups = defaultdict(list)
            
            for path in tqdm(contract_paths, desc="Version detection"):
                version = self._detect_solidity_version(Path(path))
                version_groups[version].append(path)
            
            # Separate skipped from processable
            processable_groups = {}
            skipped_results = []
            
            for version, paths in version_groups.items():
                if self._should_skip_version(version):
                    # Create failed results for skipped
                    for path in paths:
                        skipped_results.append(SlitherResult(
                            contract_path=path,
                            success=False,
                            error_type="skipped_ancient_version",
                            error_message=f"Skipped: Ancient/unsupported version {version}",
                            detected_version=version
                        ))
                else:
                    processable_groups[version] = paths
            
            print(f"\n✅ Found {len(version_groups)} versions")
            print(f"⚠️  Skipped {len(version_groups) - len(processable_groups)} ancient versions ({len(skipped_results)} contracts)")
            print(f"🚀 Processing {len(processable_groups)} version groups in PARALLEL\n")
            
            # PHASE 2: PARALLEL VERSION PROCESSING (Optimization E)
            print("="*80)
            print("⚡ Phase 2: PARALLEL VERSION GROUP PROCESSING")
            print("="*80 + "\n")
            
            all_results = []
            
            # Process version groups in parallel!
            # Use ThreadPoolExecutor with limited workers to avoid overload
            max_version_workers = min(len(processable_groups), 4)  # Max 4 versions at once
            
            with ThreadPoolExecutor(max_workers=max_version_workers) as version_executor:
                version_futures = {}
                
                for version, paths in processable_groups.items():
                    future = version_executor.submit(
                        self._process_version_group_turbo,
                        paths,
                        version,
                        max_workers,
                        context
                    )
                    version_futures[future] = (version, len(paths))
                
                # Collect results as they complete
                for future in as_completed(version_futures):
                    version, count = version_futures[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                        successful = sum(1 for r in results if r.success)
                        print(f"✅ v{version}: {successful}/{count} successful")
                    except Exception as e:
                        logger.error(f"Version {version} failed: {e}")
                        # Create failed results
                        for path in processable_groups[version]:
                            all_results.append(SlitherResult(
                                contract_path=path,
                                success=False,
                                error_type="version_group_error",
                                error_message=str(e),
                                detected_version=version
                            ))
            
            # Add skipped results
            all_results.extend(skipped_results)
            
            # PHASE 3: Summary and save
            self._print_summary(all_results)
            
            if save_dir:
                self._save_results(all_results, save_dir)
                self._save_failure_report(all_results, save_dir)
            
            return all_results
            
        finally:
            # Cleanup Docker
            if context == ExecutionContext.DOCKER:
                self._stop_docker_container()
    
    def _process_version_group_turbo(
        self,
        contract_paths: List[str],
        version: str,
        max_workers: int,
        context: ExecutionContext
    ) -> List[SlitherResult]:
        """
        Process a version group with enhanced error handling for turbo mode.
        
        This is called in parallel for each version group!
        """
        print(f"\n📦 Processing {len(contract_paths)} contracts with Solidity v{version}...")
        
        # Install/switch version
        if context == ExecutionContext.DOCKER:
            if not self._switch_version_docker(version):
                print(f"⚠️  Version {version} not available in Docker, using local")
                context = ExecutionContext.LOCAL
        
        if context == ExecutionContext.LOCAL:
            if self.auto_install and not self._ensure_solc_version(version):
                logger.warning(f"Failed to install {version}")
                return [
                    SlitherResult(
                        contract_path=p,
                        success=False,
                        error_type="version_install_failed",
                        error_message=f"Failed to install {version}",
                        detected_version=version
                    )
                    for p in contract_paths
                ]
        
        # Use parent's _process_version_group (it already has parallel processing)
        return self._process_version_group(contract_paths, version, max_workers, context)

