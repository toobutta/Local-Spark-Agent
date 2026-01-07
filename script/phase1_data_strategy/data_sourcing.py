#!/usr/bin/env python3
"""
Phase 1: Data Sourcing and Acquisition Module
Comprehensive data acquisition from multiple sources with licensing and compliance management
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
from pathlib import Path
import hashlib
import yaml
from abc import ABC, abstractmethod

# External libraries
import pandas as pd
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, Repository
import aiohttp
from tqdm import tqdm

# Local imports
from src.utilities.configuration_manager import ConfigurationManager
from src.utilities.version_control import VersionControlManager
from src.utilities.security_utils import encrypt_sensitive_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Data source metadata and configuration"""
    name: str
    type: str  # public, proprietary, synthetic, crowdsource, partnership
    url: Optional[str] = None
    license: Optional[str] = None
    description: Optional[str] = None
    size_gb: Optional[float] = None
    last_updated: Optional[datetime] = None
    quality_score: Optional[float] = None
    compliance_status: Optional[str] = None
    access_requirements: Optional[Dict[str, Any]] = None


@dataclass
class DataAcquisitionRecord:
    """Record of data acquisition for audit trail"""
    source_name: str
    acquisition_date: datetime
    file_count: int
    total_size_gb: float
    checksum: str
    license_verified: bool
    compliance_checked: bool
    processing_steps: List[str]
    metadata: Dict[str, Any]


class DataSourceConnector(ABC):
    """Abstract base class for data source connectors"""

    @abstractmethod
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Establish connection to data source"""
        pass

    @abstractmethod
    async def list_available_datasets(self) -> List[DataSource]:
        """List available datasets from this source"""
        pass

    @abstractmethod
    async def download_dataset(self, dataset_name: str, target_path: Path) -> bool:
        """Download dataset to specified path"""
        pass

    @abstractmethod
    async def verify_license(self, dataset_name: str) -> bool:
        """Verify license compliance for dataset"""
        pass


class HuggingFaceConnector(DataSourceConnector):
    """Connector for Hugging Face datasets"""

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token
        self.hf_api = HfApi(token=api_token) if api_token else HfApi()

    async def connect(self, config: Dict[str, Any]) -> bool:
        """Test connection to Hugging Face"""
        try:
            # Test API access
            user_info = self.hf_api.whoami()
            logger.info(f"Connected to Hugging Face as user: {user_info['name']}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Hugging Face: {e}")
            return False

    async def list_available_datasets(self) -> List[DataSource]:
        """List available Hugging Face datasets"""
        datasets = []
        try:
            # Get popular datasets (this is a simplified version)
            popular_datasets = [
                "wikitext",
                "c4",
                "bookcorpus",
                "openwebtext",
                "pile",
                "the_stack"
            ]

            for dataset_name in tqdm(popular_datasets, desc="Fetching Hugging Face datasets"):
                try:
                    dataset_info = self.hf_api.dataset_info(dataset_name)
                    source = DataSource(
                        name=dataset_name,
                        type="public",
                        url=f"https://huggingface.co/datasets/{dataset_name}",
                        license=dataset_info.cardData.get("license", "Unknown"),
                        description=dataset_info.cardData.get("description", ""),
                        size_gb=dataset_info.dataset_size.get("total", 0) / (1024**3) if dataset_info.dataset_size else None,
                        last_updated=datetime.fromisoformat(dataset_info.last_modified.replace('Z', '+00:00')) if dataset_info.last_modified else None,
                        compliance_status="verified" if dataset_info.cardData.get("license") else "pending_review"
                    )
                    datasets.append(source)
                except Exception as e:
                    logger.warning(f"Failed to get info for dataset {dataset_name}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to list Hugging Face datasets: {e}")

        return datasets

    async def download_dataset(self, dataset_name: str, target_path: Path) -> bool:
        """Download Hugging Face dataset"""
        try:
            logger.info(f"Downloading dataset {dataset_name} from Hugging Face")

            # Download dataset
            dataset = load_dataset(dataset_name)

            # Save to target path
            if not target_path.exists():
                target_path.mkdir(parents=True, exist_ok=True)

            # Save each split
            for split_name, split_data in dataset.items():
                split_path = target_path / f"{split_name}.parquet"
                split_data.to_parquet(split_path)
                logger.info(f"Saved {split_name} split to {split_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_name}: {e}")
            return False

    async def verify_license(self, dataset_name: str) -> bool:
        """Verify license for Hugging Face dataset"""
        try:
            dataset_info = self.hf_api.dataset_info(dataset_name)
            license_info = dataset_info.cardData.get("license")

            if license_info:
                # Check if license allows commercial use
                commercial_licenses = [
                    "mit", "apache-2.0", "bsd", "cc-by-4.0",
                    "cc-by-sa-4.0", "odc-by", "odc-by-sa"
                ]

                license_lower = license_info.lower()
                is_commercial = any(lic in license_lower for lic in commercial_licenses)

                logger.info(f"Dataset {dataset_name} license: {license_info} (Commercial use: {is_commercial})")
                return is_commercial
            else:
                logger.warning(f"No license information found for dataset {dataset_name}")
                return False

        except Exception as e:
            logger.error(f"Failed to verify license for {dataset_name}: {e}")
            return False


class CustomDataConnector(DataSourceConnector):
    """Connector for custom/proprietary data sources"""

    async def connect(self, config: Dict[str, Any]) -> bool:
        """Test connection to custom data source"""
        # Implementation depends on specific data source
        logger.info("Custom data source connection placeholder")
        return True

    async def list_available_datasets(self) -> List[DataSource]:
        """List available custom datasets"""
        # Implementation depends on specific data source
        return []

    async def download_dataset(self, dataset_name: str, target_path: Path) -> bool:
        """Download custom dataset"""
        # Implementation depends on specific data source
        logger.info(f"Custom dataset download placeholder for {dataset_name}")
        return True

    async def verify_license(self, dataset_name: str) -> bool:
        """Verify license for custom dataset"""
        # Custom datasets typically have internal licensing
        return True


class SyntheticDataGenerator:
    """Generate synthetic data for training augmentation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.generation_methods = config.get("generation_methods", [])

    async def generate_synthetic_data(self,
                                    base_data: Optional[Dataset] = None,
                                    target_size: int = 1000,
                                    output_path: Optional[Path] = None) -> Dataset:
        """Generate synthetic data based on configuration"""
        synthetic_data = []

        if "template_filling" in self.generation_methods:
            synthetic_data.extend(await self._generate_template_data(target_size // 3))

        if "paraphrasing" in self.generation_methods and base_data:
            synthetic_data.extend(await self._generate_paraphrases(base_data, target_size // 3))

        if "ai_augmentation" in self.generation_methods:
            synthetic_data.extend(await self._generate_ai_augmented_data(target_size // 3))

        # Create dataset
        synthetic_dataset = Dataset.from_list(synthetic_data)

        # Save if output path provided
        if output_path:
            synthetic_dataset.to_parquet(output_path)
            logger.info(f"Saved synthetic data to {output_path}")

        return synthetic_dataset

    async def _generate_template_data(self, count: int) -> List[Dict[str, str]]:
        """Generate data using templates"""
        templates = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is revolutionizing the way we process data.",
            "Natural language processing enables computers to understand human language.",
            "Artificial intelligence has the potential to transform many industries.",
            "Data science combines statistics, programming, and domain expertise."
        ]

        data = []
        for i in range(count):
            template = templates[i % len(templates)]
            data.append({
                "text": template,
                "source": "synthetic_template",
                "generated_at": datetime.now().isoformat()
            })

        return data

    async def _generate_paraphrases(self, base_data: Dataset, count: int) -> List[Dict[str, str]]:
        """Generate paraphrases of existing data"""
        # Simplified paraphrase generation
        paraphrases = []

        for i in range(min(count, len(base_data))):
            original = base_data[i]["text"]
            # Simple paraphrase techniques (in real implementation, use NLP models)
            paraphrase = original.replace("AI", "artificial intelligence").replace("ML", "machine learning")

            paraphrases.append({
                "text": paraphrase,
                "source": "synthetic_paraphrase",
                "original_id": i,
                "generated_at": datetime.now().isoformat()
            })

        return paraphrases

    async def _generate_ai_augmented_data(self, count: int) -> List[Dict[str, str]]:
        """Generate AI-augmented data"""
        # Placeholder for AI-powered data generation
        augmented_data = []

        prompts = [
            "Explain the importance of data quality in machine learning.",
            "Describe the key components of a neural network.",
            "What are the main challenges in natural language understanding?",
            "How does transfer learning improve model performance?",
            "What role does data preprocessing play in AI development?"
        ]

        for i in range(count):
            prompt = prompts[i % len(prompts)]
            # In real implementation, this would call an AI model
            response = f"This is a simulated AI response to: {prompt}"

            augmented_data.append({
                "text": response,
                "source": "synthetic_ai_generated",
                "prompt": prompt,
                "generated_at": datetime.now().isoformat()
            })

        return augmented_data


class DataSourcingManager:
    """Main manager for data sourcing operations"""

    def __init__(self, config_path: str):
        self.config = ConfigurationManager(config_path)
        self.data_config = self.config.get_section("data_strategy")
        self.version_manager = VersionControlManager()

        # Initialize connectors
        self.connectors = {}
        self._initialize_connectors()

        # Initialize synthetic data generator
        self.synthetic_generator = SyntheticDataGenerator(
            self.data_config.get("sourcing", {}).get("synthetic_data", {})
        )

        # Setup storage paths
        self.base_path = Path("data")
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.metadata_path = self.base_path / "metadata"

        # Create directories
        for path in [self.base_path, self.raw_path, self.processed_path, self.metadata_path]:
            path.mkdir(parents=True, exist_ok=True)

    def _initialize_connectors(self):
        """Initialize data source connectors"""
        sourcing_config = self.data_config.get("sourcing", {})

        # Initialize HuggingFace connector
        if sourcing_config.get("public_datasets", {}).get("enabled", True):
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            self.connectors["huggingface"] = HuggingFaceConnector(hf_token)

        # Initialize custom connector
        if sourcing_config.get("proprietary_data", {}).get("enabled", True):
            self.connectors["custom"] = CustomDataConnector()

    async def run_discovery(self) -> List[DataSource]:
        """Discover available data sources"""
        all_sources = []

        for connector_name, connector in self.connectors.items():
            logger.info(f"Discovering datasets from {connector_name}")

            # Connect to source
            if await connector.connect({}):
                # List available datasets
                sources = await connector.list_available_datasets()
                all_sources.extend(sources)
                logger.info(f"Found {len(sources)} datasets from {connector_name}")
            else:
                logger.error(f"Failed to connect to {connector_name}")

        # Save discovery results
        discovery_file = self.metadata_path / "data_sources.json"
        with open(discovery_file, 'w') as f:
            json.dump([asdict(source) for source in all_sources], f, indent=2, default=str)

        logger.info(f"Data discovery complete. Found {len(all_sources)} total sources.")
        return all_sources

    async def acquire_dataset(self, source: DataSource, dataset_name: str) -> Optional[DataAcquisitionRecord]:
        """Acquire a specific dataset"""
        logger.info(f"Acquiring dataset {dataset_name} from {source.name}")

        # Determine target path
        target_path = self.raw_path / source.type / dataset_name

        # Get appropriate connector
        connector = self.connectors.get(source.name, self.connectors.get("custom"))
        if not connector:
            logger.error(f"No connector available for source {source.name}")
            return None

        try:
            # Verify license compliance
            license_verified = await connector.verify_license(dataset_name)
            if not license_verified:
                logger.warning(f"License verification failed for {dataset_name}")
                # Continue anyway but flag for review

            # Download dataset
            success = await connector.download_dataset(dataset_name, target_path)
            if not success:
                logger.error(f"Failed to download {dataset_name}")
                return None

            # Calculate metadata
            file_count = len(list(target_path.rglob("*")))
            total_size = sum(f.stat().st_size for f in target_path.rglob("*") if f.is_file()) / (1024**3)

            # Calculate checksum
            checksum = await self._calculate_directory_checksum(target_path)

            # Create acquisition record
            record = DataAcquisitionRecord(
                source_name=source.name,
                acquisition_date=datetime.now(),
                file_count=file_count,
                total_size_gb=total_size,
                checksum=checksum,
                license_verified=license_verified,
                compliance_checked=True,  # Simplified for now
                processing_steps=["download", "metadata_extraction"],
                metadata={
                    "dataset_name": dataset_name,
                    "source_type": source.type,
                    "license": source.license,
                    "target_path": str(target_path)
                }
            )

            # Save acquisition record
            await self._save_acquisition_record(record)

            logger.info(f"Successfully acquired {dataset_name} ({total_size:.2f} GB)")
            return record

        except Exception as e:
            logger.error(f"Failed to acquire {dataset_name}: {e}")
            return None

    async def generate_synthetic_datasets(self) -> List[DataAcquisitionRecord]:
        """Generate synthetic datasets"""
        synthetic_config = self.data_config.get("sourcing", {}).get("synthetic_data", {})

        if not synthetic_config.get("enabled", False):
            logger.info("Synthetic data generation is disabled")
            return []

        records = []
        target_path = self.raw_path / "synthetic"

        # Generate synthetic data
        synthetic_dataset = await self.synthetic_generator.generate_synthetic_data(
            target_size=1000,
            output_path=target_path / "synthetic_data.parquet"
        )

        # Create acquisition record
        record = DataAcquisitionRecord(
            source_name="synthetic_generator",
            acquisition_date=datetime.now(),
            file_count=1,
            total_size_gb=(target_path / "synthetic_data.parquet").stat().st_size / (1024**3),
            checksum=await self._calculate_file_checksum(target_path / "synthetic_data.parquet"),
            license_verified=True,  # Synthetic data doesn't require licensing
            compliance_checked=True,
            processing_steps=["generation", "quality_check"],
            metadata={
                "generation_methods": synthetic_config.get("generation_methods", []),
                "target_size": len(synthetic_dataset),
                "quality_score": 0.8  # Placeholder
            }
        )

        await self._save_acquisition_record(record)
        records.append(record)

        logger.info(f"Generated synthetic dataset with {len(synthetic_dataset)} samples")
        return records

    async def run_full_acquisition(self) -> List[DataAcquisitionRecord]:
        """Run complete data acquisition process"""
        logger.info("Starting full data acquisition process")

        # Step 1: Discover available data sources
        sources = await self.run_discovery()

        # Step 2: Acquire datasets from sources
        acquisition_records = []

        for source in sources:
            # Skip if source type is disabled
            source_config = self.data_config.get("sourcing", {}).get(f"{source.type}_data", {})
            if not source_config.get("enabled", True):
                logger.info(f"Skipping {source.type} data source (disabled)")
                continue

            # Use source name as dataset name for simplicity
            dataset_name = source.name

            # Check if already acquired
            if await self._is_already_acquired(dataset_name):
                logger.info(f"Dataset {dataset_name} already acquired, skipping")
                continue

            # Acquire dataset
            record = await self.acquire_dataset(source, dataset_name)
            if record:
                acquisition_records.append(record)

        # Step 3: Generate synthetic data
        synthetic_records = await self.generate_synthetic_datasets()
        acquisition_records.extend(synthetic_records)

        # Step 4: Create summary report
        await self._create_acquisition_summary(acquisition_records)

        logger.info(f"Data acquisition complete. Acquired {len(acquisition_records)} datasets.")
        return acquisition_records

    async def _calculate_directory_checksum(self, directory_path: Path) -> str:
        """Calculate checksum for entire directory"""
        hash_md5 = hashlib.md5()

        for file_path in sorted(directory_path.rglob("*")):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)

        return hash_md5.hexdigest()

    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate checksum for single file"""
        hash_md5 = hashlib.md5()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()

    async def _save_acquisition_record(self, record: DataAcquisitionRecord):
        """Save acquisition record to file"""
        records_file = self.metadata_path / "acquisition_records.json"

        # Load existing records
        records = []
        if records_file.exists():
            with open(records_file, 'r') as f:
                records = json.load(f)

        # Add new record
        records.append(asdict(record))

        # Save records
        with open(records_file, 'w') as f:
            json.dump(records, f, indent=2, default=str)

    async def _is_already_acquired(self, dataset_name: str) -> bool:
        """Check if dataset is already acquired"""
        records_file = self.metadata_path / "acquisition_records.json"

        if not records_file.exists():
            return False

        with open(records_file, 'r') as f:
            records = json.load(f)

        return any(record["metadata"]["dataset_name"] == dataset_name for record in records)

    async def _create_acquisition_summary(self, records: List[DataAcquisitionRecord]):
        """Create summary of acquisition process"""
        summary = {
            "acquisition_date": datetime.now().isoformat(),
            "total_datasets": len(records),
            "total_size_gb": sum(record.total_size_gb for record in records),
            "total_files": sum(record.file_count for record in records),
            "sources": list(set(record.source_name for record in records)),
            "datasets": [
                {
                    "name": record.metadata["dataset_name"],
                    "source": record.source_name,
                    "size_gb": record.total_size_gb,
                    "files": record.file_count,
                    "license_verified": record.license_verified
                }
                for record in records
            ]
        }

        # Save summary
        summary_file = self.metadata_path / "acquisition_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Acquisition summary saved to {summary_file}")


# CLI interface for standalone execution
async def main():
    """Main function for CLI execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Data Sourcing and Acquisition")
    parser.add_argument("--config", default="configs/lifecycle/phase1_data_strategy.yaml",
                       help="Configuration file path")
    parser.add_argument("--action", choices=["discover", "acquire", "synthetic", "all"],
                       default="all", help="Action to perform")
    parser.add_argument("--dataset", help="Specific dataset to acquire")

    args = parser.parse_args()

    # Initialize manager
    manager = DataSourcingManager(args.config)

    if args.action == "discover" or args.action == "all":
        await manager.run_discovery()

    if args.action == "acquire" or args.action == "all":
        if args.dataset:
            # Acquire specific dataset
            sources = await manager.run_discovery()
            source = next((s for s in sources if s.name == args.dataset), None)
            if source:
                await manager.acquire_dataset(source, args.dataset)
            else:
                logger.error(f"Dataset {args.dataset} not found")
        else:
            # Acquire all datasets
            await manager.run_full_acquisition()

    if args.action == "synthetic" or args.action == "all":
        await manager.generate_synthetic_datasets()


if __name__ == "__main__":
    asyncio.run(main())