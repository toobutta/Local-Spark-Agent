#!/usr/bin/env python3
"""
Phase 2: Model Conception & Design - Requirement Analysis Module
Comprehensive stakeholder requirement gathering and analysis framework
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from src.utilities.configuration_manager import ConfigurationManager
from src.utilities.version_control import VersionControlManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StakeholderType(Enum):
    """Enumeration of stakeholder types"""
    BUSINESS_STAKEHOLDER = "business_stakeholder"
    TECHNICAL_STAKEHOLDER = "technical_stakeholder"
    END_USER = "end_user"
    REGULATORY_BODY = "regulatory_body"
    ETHICAL_REVIEWER = "ethical_reviewer"


class RequirementPriority(Enum):
    """Enumeration of requirement priorities"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RequirementType(Enum):
    """Enumeration of requirement types"""
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    CONSTRAINT = "constraint"
    ASSUMPTION = "assumption"


@dataclass
class Stakeholder:
    """Stakeholder information and configuration"""
    name: str
    role: str
    stakeholder_type: StakeholderType
    concerns: List[str]
    success_metrics: List[str]
    communication_frequency: str
    contact_info: Optional[str] = None
    influence_level: float = 0.5  # 0-1 scale
    interest_level: float = 0.5  # 0-1 scale
    availability: Optional[str] = None
    time_zone: Optional[str] = None
    language: str = "English"
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Requirement:
    """Individual requirement with full metadata"""
    id: str
    title: str
    description: str
    requirement_type: RequirementType
    priority: RequirementPriority
    stakeholder: str
    category: str
    acceptance_criteria: List[str]
    success_metrics: List[str]
    dependencies: List[str] = field(default_factory=list)
    rationale: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    verification_method: Optional[str] = None
    estimated_effort: Optional[str] = None
    risk_level: str = "medium"
    status: str = "proposed"
    created_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequirementAnalysis:
    """Complete requirement analysis results"""
    project_name: str
    analysis_date: datetime
    stakeholders: List[Stakeholder]
    requirements: List[Requirement]
    requirement_categories: List[str]
    priority_distribution: Dict[str, int]
    stakeholder_analysis: Dict[str, Any]
    dependency_matrix: Dict[str, List[str]]
    risk_assessment: Dict[str, Any]
    gap_analysis: Dict[str, Any]
    recommendations: List[str]
    next_steps: List[str]


class RequirementAnalyzer:
    """Analyzes requirements for consistency, completeness, and feasibility"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stakeholders: List[Stakeholder] = []
        self.requirements: List[Requirement] = []

    def add_stakeholder(self, stakeholder: Stakeholder):
        """Add a stakeholder to the analysis"""
        self.stakeholders.append(stakeholder)
        logger.info(f"Added stakeholder: {stakeholder.name} ({stakeholder.role})")

    def add_requirement(self, requirement: Requirement):
        """Add a requirement to the analysis"""
        self.requirements.append(requirement)
        logger.info(f"Added requirement: {requirement.title}")

    def analyze_consistency(self) -> Dict[str, Any]:
        """Analyze requirement consistency and identify conflicts"""
        conflicts = []
        redundancies = []
        gaps = []

        # Check for conflicts
        for i, req1 in enumerate(self.requirements):
            for j, req2 in enumerate(self.requirements[i+1:], i+1):
                if self._check_conflict(req1, req2):
                    conflicts.append({
                        "requirement1": req1.id,
                        "requirement2": req2.id,
                        "conflict_type": "functional_conflict",
                        "description": f"Requirements {req1.id} and {req2.id} may conflict"
                    })

        # Check for redundancies
        for i, req1 in enumerate(self.requirements):
            for j, req2 in enumerate(self.requirements[i+1:], i+1):
                if self._check_redundancy(req1, req2):
                    redundancies.append({
                        "requirement1": req1.id,
                        "requirement2": req2.id,
                        "similarity_score": self._calculate_similarity(req1, req2),
                        "recommendation": "Consider merging or clarifying the distinction"
                    })

        # Check for gaps
        categories = set(req.category for req in self.requirements)
        expected_categories = set(self.config.get("expected_categories", []))
        missing_categories = expected_categories - categories

        for category in missing_categories:
            gaps.append({
                "category": category,
                "description": f"No requirements found for category: {category}",
                "recommendation": "Consider adding requirements for this category"
            })

        return {
            "conflicts": conflicts,
            "redundancies": redundancies,
            "gaps": gaps,
            "total_conflicts": len(conflicts),
            "total_redundancies": len(redundancies),
            "total_gaps": len(gaps)
        }

    def analyze_completeness(self) -> Dict[str, Any]:
        """Analyze requirement completeness"""
        completeness_scores = {}
        missing_elements = []

        # Check each requirement for completeness
        for req in self.requirements:
            score = 0
            total_checks = 0

            # Check for essential elements
            if req.description and len(req.description) > 50:
                score += 1
            total_checks += 1

            if req.acceptance_criteria:
                score += 1
            total_checks += 1

            if req.success_metrics:
                score += 1
            total_checks += 1

            if req.verification_method:
                score += 1
            total_checks += 1

            if req.rationale:
                score += 1
            total_checks += 1

            completeness_score = score / total_checks if total_checks > 0 else 0
            completeness_scores[req.id] = completeness_score

            if completeness_score < 0.8:
                missing_elements.append({
                    "requirement_id": req.id,
                    "score": completeness_score,
                    "missing_items": self._get_missing_elements(req)
                })

        # Calculate overall completeness
        overall_score = np.mean(list(completeness_scores.values())) if completeness_scores else 0

        return {
            "overall_completeness_score": overall_score,
            "requirement_scores": completeness_scores,
            "incomplete_requirements": missing_elements,
            "total_requirements": len(self.requirements),
            "complete_requirements": len([r for r, s in completeness_scores.items() if s >= 0.8])
        }

    def analyze_feasibility(self) -> Dict[str, Any]:
        """Analyze technical and business feasibility"""
        feasibility_scores = {}
        feasibility_risks = []

        for req in self.requirements:
            # Analyze technical feasibility
            technical_score = self._assess_technical_feasibility(req)

            # Analyze business feasibility
            business_score = self._assess_business_feasibility(req)

            # Analyze resource feasibility
            resource_score = self._assess_resource_feasibility(req)

            # Calculate overall feasibility
            overall_score = (technical_score + business_score + resource_score) / 3
            feasibility_scores[req.id] = overall_score

            # Identify risks
            if overall_score < 0.6:
                feasibility_risks.append({
                    "requirement_id": req.id,
                    "overall_score": overall_score,
                    "technical_score": technical_score,
                    "business_score": business_score,
                    "resource_score": resource_score,
                    "risk_factors": self._identify_risk_factors(req),
                    "recommendations": self._generate_feasibility_recommendations(req)
                })

        return {
            "feasibility_scores": feasibility_scores,
            "high_risk_requirements": feasibility_risks,
            "average_feasibility": np.mean(list(feasibility_scores.values())) if feasibility_scores else 0,
            "total_requirements": len(self.requirements),
            "feasible_requirements": len([r for r, s in feasibility_scores.items() if s >= 0.6])
        }

    def prioritize_requirements(self) -> Dict[str, Any]:
        """Prioritize requirements based on multiple criteria"""
        priority_scores = {}

        for req in self.requirements:
            # Base priority from stakeholder influence
            stakeholder = next((s for s in self.stakeholders if s.name == req.stakeholder), None)
            influence_score = stakeholder.influence_level if stakeholder else 0.5

            # Priority weight mapping
            priority_weights = {
                RequirementPriority.CRITICAL: 1.0,
                RequirementPriority.HIGH: 0.8,
                RequirementPriority.MEDIUM: 0.6,
                RequirementPriority.LOW: 0.4
            }
            priority_weight = priority_weights.get(req.priority, 0.5)

            # Risk factor (lower risk = higher priority)
            risk_weights = {"low": 1.0, "medium": 0.8, "high": 0.6}
            risk_weight = risk_weights.get(req.risk_level, 0.8)

            # Effort factor (lower effort = higher priority)
            effort_weights = {"low": 1.0, "medium": 0.8, "high": 0.6}
            effort_weight = effort_weights.get(req.estimated_effort, 0.8) if req.estimated_effort else 0.8

            # Calculate composite priority score
            composite_score = (priority_weight * 0.4 + influence_score * 0.3 +
                            risk_weight * 0.2 + effort_weight * 0.1)

            priority_scores[req.id] = composite_score

        # Sort by priority score
        sorted_requirements = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "priority_scores": priority_scores,
            "sorted_requirements": sorted_requirements,
            "top_priorities": sorted_requirements[:10],  # Top 10 priorities
            "priority_matrix": self._create_priority_matrix()
        }

    def _check_conflict(self, req1: Requirement, req2: Requirement) -> bool:
        """Check if two requirements conflict"""
        # Simplified conflict detection
        conflict_keywords = ["not", "never", "avoid", "prevent", "restrict", "limit"]

        # Check if requirements have opposite goals
        if any(keyword in req1.description.lower() for keyword in conflict_keywords):
            if any(keyword in req2.description.lower() for keyword in conflict_keywords):
                # More sophisticated conflict detection would go here
                return False  # Placeholder

        return False

    def _check_redundancy(self, req1: Requirement, req2: Requirement) -> bool:
        """Check if two requirements are redundant"""
        similarity = self._calculate_similarity(req1, req2)
        return similarity > 0.8

    def _calculate_similarity(self, req1: Requirement, req2: Requirement) -> float:
        """Calculate similarity between two requirements"""
        # Simple keyword-based similarity (more sophisticated methods could be used)
        words1 = set(req1.description.lower().split())
        words2 = set(req2.description.lower().split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0

    def _get_missing_elements(self, req: Requirement) -> List[str]:
        """Identify missing elements in a requirement"""
        missing = []
        if not req.description or len(req.description) < 50:
            missing.append("detailed_description")
        if not req.acceptance_criteria:
            missing.append("acceptance_criteria")
        if not req.success_metrics:
            missing.append("success_metrics")
        if not req.verification_method:
            missing.append("verification_method")
        if not req.rationale:
            missing.append("rationale")
        return missing

    def _assess_technical_feasibility(self, req: Requirement) -> float:
        """Assess technical feasibility of requirement"""
        # Simplified technical feasibility assessment
        base_score = 0.8

        # Adjust based on requirement complexity
        if "complex" in req.description.lower() or req.estimated_effort == "high":
            base_score -= 0.2

        # Adjust based on requirement type
        if req.requirement_type == RequirementType.CONSTRAINT:
            base_score += 0.1

        # Adjust based on risk level
        if req.risk_level == "high":
            base_score -= 0.3
        elif req.risk_level == "low":
            base_score += 0.1

        return max(0, min(1, base_score))

    def _assess_business_feasibility(self, req: Requirement) -> float:
        """Assess business feasibility of requirement"""
        # Simplified business feasibility assessment
        stakeholder = next((s for s in self.stakeholders if s.name == req.stakeholder), None)

        if not stakeholder:
            return 0.5

        # Business value based on stakeholder influence and interest
        business_value = (stakeholder.influence_level + stakeholder.interest_level) / 2

        # Adjust based on requirement priority
        priority_values = {
            RequirementPriority.CRITICAL: 1.0,
            RequirementPriority.HIGH: 0.8,
            RequirementPriority.MEDIUM: 0.6,
            RequirementPriority.LOW: 0.4
        }
        priority_value = priority_values.get(req.priority, 0.5)

        return (business_value + priority_value) / 2

    def _assess_resource_feasibility(self, req: Requirement) -> float:
        """Assess resource feasibility of requirement"""
        # Simplified resource feasibility assessment
        base_score = 0.7

        # Adjust based on estimated effort
        effort_scores = {"low": 1.0, "medium": 0.8, "high": 0.6}
        if req.estimated_effort:
            effort_score = effort_scores.get(req.estimated_effort, 0.8)
            base_score = (base_score + effort_score) / 2

        # Adjust based on dependencies
        if len(req.dependencies) > 3:
            base_score -= 0.2

        return max(0, min(1, base_score))

    def _identify_risk_factors(self, req: Requirement) -> List[str]:
        """Identify risk factors for a requirement"""
        risks = []

        if req.risk_level == "high":
            risks.append("high_risk_level")

        if len(req.dependencies) > 5:
            risks.append("many_dependencies")

        if req.estimated_effort == "high":
            risks.append("high_effort")

        if not req.acceptance_criteria:
            risks.append("unclear_acceptance")

        if not req.verification_method:
            risks.append("difficult_to_verify")

        return risks

    def _generate_feasibility_recommendations(self, req: Requirement) -> List[str]:
        """Generate feasibility improvement recommendations"""
        recommendations = []

        if req.risk_level == "high":
            recommendations.append("Conduct detailed risk assessment and mitigation planning")

        if req.estimated_effort == "high":
            recommendations.append("Consider breaking down into smaller, manageable requirements")

        if len(req.dependencies) > 5:
            recommendations.append("Review and minimize dependencies")

        if not req.acceptance_criteria:
            recommendations.append("Define clear acceptance criteria")

        if not req.verification_method:
            recommendations.append("Establish verification and testing methods")

        return recommendations

    def _create_priority_matrix(self) -> Dict[str, Any]:
        """Create a priority matrix for requirements"""
        matrix = defaultdict(lambda: defaultdict(list))

        for req in self.requirements:
            matrix[req.priority.value][req.requirement_type.value].append(req.id)

        return dict(matrix)


class StakeholderAnalyzer:
    """Analyzes stakeholders and their influence/interest"""

    def __init__(self, stakeholders: List[Stakeholder]):
        self.stakeholders = stakeholders

    def create_stakeholder_map(self) -> Dict[str, Any]:
        """Create stakeholder influence/interest map"""
        stakeholder_data = []

        for stakeholder in self.stakeholders:
            stakeholder_data.append({
                "name": stakeholder.name,
                "role": stakeholder.role,
                "type": stakeholder.stakeholder_type.value,
                "influence": stakeholder.influence_level,
                "interest": stakeholder.interest_level,
                "contact": stakeholder.contact_info,
                "availability": stakeholder.availability,
                "timezone": stakeholder.time_zone
            })

        # Categorize stakeholders
        manage_closely = [s for s in self.stakeholders if s.influence_level >= 0.7 and s.interest_level >= 0.7]
        keep_satisfied = [s for s in self.stakeholders if s.influence_level >= 0.7 and s.interest_level < 0.7]
        keep_informed = [s for s in self.stakeholders if s.influence_level < 0.7 and s.interest_level >= 0.7]
        monitor = [s for s in self.stakeholders if s.influence_level < 0.7 and s.interest_level < 0.7]

        return {
            "stakeholders": stakeholder_data,
            "categories": {
                "manage_closely": [s.name for s in manage_closely],
                "keep_satisfied": [s.name for s in keep_satisfied],
                "keep_informed": [s.name for s in keep_informed],
                "monitor": [s.name for s in monitor]
            },
            "total_stakeholders": len(self.stakeholders),
            "high_influence": len([s for s in self.stakeholders if s.influence_level >= 0.7]),
            "high_interest": len([s for s in self.stakeholders if s.interest_level >= 0.7])
        }

    def analyze_engagement_strategy(self) -> Dict[str, Any]:
        """Analyze stakeholder engagement strategies"""
        strategies = {}

        for stakeholder in self.stakeholders:
            strategy = self._determine_engagement_strategy(stakeholder)
            strategies[stakeholder.name] = {
                "strategy": strategy["name"],
                "frequency": strategy["frequency"],
                "methods": strategy["methods"],
                "key_messages": strategy["key_messages"],
                "success_metrics": strategy["success_metrics"]
            }

        return strategies

    def _determine_engagement_strategy(self, stakeholder: Stakeholder) -> Dict[str, Any]:
        """Determine appropriate engagement strategy for stakeholder"""
        if stakeholder.influence_level >= 0.7 and stakeholder.interest_level >= 0.7:
            return {
                "name": "Partner",
                "frequency": "Weekly",
                "methods": ["Workshops", "Co-design sessions", "Regular reviews"],
                "key_messages": ["Strategic alignment", "Value creation", "Shared success"],
                "success_metrics": ["Active participation", "Decision making", "Resource commitment"]
            }
        elif stakeholder.influence_level >= 0.7:
            return {
                "name": "Consult",
                "frequency": "Bi-weekly",
                "methods": ["Interviews", "Surveys", "Review meetings"],
                "key_messages": ["Impact assessment", "Risk management", "Compliance"],
                "success_metrics": ["Feedback quality", "Issue resolution", "Support level"]
            }
        elif stakeholder.interest_level >= 0.7:
            return {
                "name": "Involve",
                "frequency": "Monthly",
                "methods": ["Focus groups", "User testing", "Feedback sessions"],
                "key_messages": ["User experience", "Feature requirements", "Usability"],
                "success_metrics": ["User satisfaction", "Feature adoption", "Engagement quality"]
            }
        else:
            return {
                "name": "Inform",
                "frequency": "Quarterly",
                "methods": ["Newsletters", "Updates", "Reports"],
                "key_messages": ["Progress updates", "General information", "High-level outcomes"],
                "success_metrics": ["Awareness level", "Information retention", "Satisfaction"]
            }


class RequirementAnalysisManager:
    """Main manager for requirement analysis process"""

    def __init__(self, config_path: str):
        self.config = ConfigurationManager(config_path)
        self.requirement_config = self.config.get_section("model_conception")
        self.version_manager = VersionControlManager()

        # Initialize analyzers
        self.requirement_analyzer = RequirementAnalyzer(self.requirement_config)
        self.stakeholder_analyzer = None  # Will be initialized after stakeholders are added

        # Setup output paths
        self.output_path = Path("docs/phase2_conception_design")
        self.output_path.mkdir(parents=True, exist_ok=True)

    def load_stakeholders_from_config(self) -> List[Stakeholder]:
        """Load stakeholders from configuration"""
        stakeholders = []
        stakeholder_config = self.requirement_config.get("stakeholder_analysis", {})

        for category, stakeholder_list in stakeholder_config.get("stakeholders", {}).items():
            for stakeholder_info in stakeholder_list:
                stakeholder = Stakeholder(
                    name=f"{stakeholder_info['role']}_{len(stakeholders) + 1}",
                    role=stakeholder_info["role"],
                    stakeholder_type=StakeholderType(category),
                    concerns=stakeholder_info.get("concerns", []),
                    success_metrics=stakeholder_info.get("success_metrics", []),
                    communication_frequency=stakeholder_info.get("communication_frequency", "Monthly"),
                    influence_level=0.7,  # Default values
                    interest_level=0.7
                )
                stakeholders.append(stakeholder)
                self.requirement_analyzer.add_stakeholder(stakeholder)

        self.stakeholder_analyzer = StakeholderAnalyzer(stakeholders)
        return stakeholders

    def create_sample_requirements(self) -> List[Requirement]:
        """Create sample requirements based on configuration"""
        requirements = []
        tech_specs = self.requirement_config.get("technical_specifications", {})

        # Create performance requirements
        performance_targets = tech_specs.get("performance_targets", {})
        for category, targets in performance_targets.items():
            for target in targets:
                req = Requirement(
                    id=f"REQ_{category.upper()}_{len(requirements) + 1:03d}",
                    title=f"{target.get('metric', 'Unknown').title()} Requirement",
                    description=f"The system shall achieve {target.get('metric', 'performance')} of {target.get('target', 'unknown')} with tolerance of {target.get('tolerance', 'unknown')}",
                    requirement_type=RequirementType.NON_FUNCTIONAL,
                    priority=RequirementPriority.HIGH if target.get('priority') == 'High' else RequirementPriority.MEDIUM,
                    stakeholder="Technical_Lead_1",
                    category=category,
                    acceptance_criteria=[
                        f"System measurement shows {target.get('metric')} >= {target.get('target') - target.get('tolerance', 0)}",
                        f"Measurement is consistent over 1000 test runs",
                        f"Performance is maintained under normal load conditions"
                    ],
                    success_metrics=target.get("success_metrics", [target.get('metric')]),
                    verification_method="Automated performance testing",
                    estimated_effort="medium",
                    risk_level="low"
                )
                requirements.append(req)
                self.requirement_analyzer.add_requirement(req)

        # Create security requirements
        security_reqs = tech_specs.get("security_requirements", {})
        for category, requirements in security_reqs.items():
            if isinstance(requirements, list):
                for req_item in requirements:
                    req = Requirement(
                        id=f"REQ_SECURITY_{len(requirements) + 1:03d}",
                        title=f"Security {category.title()} Requirement",
                        description=f"The system shall implement {category}: {req_item}",
                        requirement_type=RequirementType.NON_FUNCTIONAL,
                        priority=RequirementPriority.CRITICAL,
                        stakeholder="Security_Officer_1",
                        category="security",
                        acceptance_criteria=[
                            f"Security audit verifies {category} implementation",
                            "Compliance check passes",
                            "Security testing shows no vulnerabilities"
                        ],
                        success_metrics=["Security compliance score", "Vulnerability count"],
                        verification_method="Security audit and penetration testing",
                        estimated_effort="high",
                        risk_level="medium"
                    )
                    requirements.append(req)
                    self.requirement_analyzer.add_requirement(req)

        return requirements

    async def run_complete_analysis(self) -> RequirementAnalysis:
        """Run complete requirement analysis"""
        logger.info("Starting complete requirement analysis")

        # Load stakeholders
        stakeholders = self.load_stakeholders_from_config()
        logger.info(f"Loaded {len(stakeholders)} stakeholders")

        # Create sample requirements
        requirements = self.create_sample_requirements()
        logger.info(f"Created {len(requirements)} requirements")

        # Run analysis
        consistency_analysis = self.requirement_analyzer.analyze_consistency()
        completeness_analysis = self.requirement_analyzer.analyze_completeness()
        feasibility_analysis = self.requirement_analyzer.analyze_feasibility()
        priority_analysis = self.requirement_analyzer.prioritize_requirements()

        # Stakeholder analysis
        stakeholder_map = self.stakeholder_analyzer.create_stakeholder_map()
        engagement_strategies = self.stakeholder_analyzer.analyze_engagement_strategy()

        # Create dependency matrix
        dependency_matrix = self._create_dependency_matrix()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            consistency_analysis, completeness_analysis, feasibility_analysis
        )

        # Create analysis result
        analysis = RequirementAnalysis(
            project_name="LLM Model Development",
            analysis_date=datetime.now(),
            stakeholders=stakeholders,
            requirements=requirements,
            requirement_categories=list(set(req.category for req in requirements)),
            priority_distribution=self._calculate_priority_distribution(),
            stakeholder_analysis={
                "stakeholder_map": stakeholder_map,
                "engagement_strategies": engagement_strategies
            },
            dependency_matrix=dependency_matrix,
            risk_assessment=feasibility_analysis,
            gap_analysis={
                "consistency_gaps": consistency_analysis["gaps"],
                "completeness_gaps": completeness_analysis["incomplete_requirements"]
            },
            recommendations=recommendations,
            next_steps=self._generate_next_steps(recommendations)
        )

        # Save analysis results
        await self._save_analysis_results(analysis)

        logger.info("Requirement analysis completed successfully")
        return analysis

    def _create_dependency_matrix(self) -> Dict[str, List[str]]:
        """Create dependency matrix for requirements"""
        dependencies = {}

        for req in self.requirement_analyzer.requirements:
            dependencies[req.id] = req.dependencies

        return dependencies

    def _calculate_priority_distribution(self) -> Dict[str, int]:
        """Calculate distribution of requirements by priority"""
        distribution = defaultdict(int)
        for req in self.requirement_analyzer.requirements:
            distribution[req.priority.value] += 1
        return dict(distribution)

    def _generate_recommendations(self, *analyses) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []

        for analysis in analyses:
            if isinstance(analysis, dict):
                if analysis.get("total_conflicts", 0) > 0:
                    recommendations.append(f"Resolve {analysis['total_conflicts']} requirement conflicts")

                if analysis.get("total_redundancies", 0) > 0:
                    recommendations.append(f"Review and potentially merge {analysis['total_redundancies']} redundant requirements")

                if analysis.get("total_gaps", 0) > 0:
                    recommendations.append(f"Address {analysis['total_gaps']} identified requirement gaps")

                if analysis.get("overall_completeness_score", 1.0) < 0.8:
                    recommendations.append("Improve requirement completeness by adding missing elements")

                if analysis.get("average_feasibility", 1.0) < 0.7:
                    recommendations.append("Address high-risk requirements and improve feasibility")

        return recommendations

    def _generate_next_steps(self, recommendations: List[str]) -> List[str]:
        """Generate next steps based on recommendations"""
        next_steps = [
            "Schedule stakeholder review meeting",
            "Prioritize requirements based on analysis results",
            "Create detailed requirement specifications",
            "Develop risk mitigation plans",
            "Establish requirement traceability matrix",
            "Plan requirement validation activities"
        ]

        # Add specific next steps based on recommendations
        if any("conflict" in rec.lower() for rec in recommendations):
            next_steps.append("Resolve requirement conflicts through stakeholder mediation")

        if any("completeness" in rec.lower() for rec in recommendations):
            next_steps.append("Complete missing requirement elements")

        if any("feasibility" in rec.lower() for rec in recommendations):
            next_steps.append("Conduct detailed feasibility studies for high-risk requirements")

        return next_steps

    async def _save_analysis_results(self, analysis: RequirementAnalysis):
        """Save analysis results to files"""
        # Save main analysis
        analysis_file = self.output_path / "requirement_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(asdict(analysis), f, indent=2, default=str)

        # Save requirements as CSV
        req_data = []
        for req in analysis.requirements:
            req_data.append({
                "ID": req.id,
                "Title": req.title,
                "Type": req.requirement_type.value,
                "Priority": req.priority.value,
                "Category": req.category,
                "Stakeholder": req.stakeholder,
                "Risk Level": req.risk_level,
                "Status": req.status
            })

        df = pd.DataFrame(req_data)
        df.to_csv(self.output_path / "requirements.csv", index=False)

        # Save analysis summary
        summary = {
            "analysis_date": analysis.analysis_date.isoformat(),
            "total_stakeholders": len(analysis.stakeholders),
            "total_requirements": len(analysis.requirements),
            "priority_distribution": analysis.priority_distribution,
            "requirement_categories": analysis.requirement_categories,
            "recommendations": analysis.recommendations,
            "next_steps": analysis.next_steps
        }

        summary_file = self.output_path / "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Analysis results saved to {self.output_path}")


# CLI interface for standalone execution
async def main():
    """Main function for CLI execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Requirement Analysis")
    parser.add_argument("--config", default="configs/lifecycle/phase2_conception_design.yaml",
                       help="Configuration file path")
    parser.add_argument("--output", default="docs/phase2_conception_design",
                       help="Output directory")

    args = parser.parse_args()

    # Initialize manager
    manager = RequirementAnalysisManager(args.config)
    if args.output:
        manager.output_path = Path(args.output)

    # Run analysis
    analysis = await manager.run_complete_analysis()

    # Print summary
    print(f"\nRequirement Analysis Summary:")
    print(f"Total Stakeholders: {len(analysis.stakeholders)}")
    print(f"Total Requirements: {len(analysis.requirements)}")
    print(f"Requirement Categories: {len(analysis.requirement_categories)}")
    print(f"Recommendations: {len(analysis.recommendations)}")
    print(f"Next Steps: {len(analysis.next_steps)}")
    print(f"\nAnalysis saved to: {manager.output_path}")


if __name__ == "__main__":
    asyncio.run(main())