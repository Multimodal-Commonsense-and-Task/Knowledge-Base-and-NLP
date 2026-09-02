"""
Taxonomy to Scenario Task
Converts taxonomy data into scenario descriptions.
"""
import json
import os
import glob
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from src.dataset_generation.task import BaseTask
from src.dataset_generation.prompters import RiskPipelinePrompter


class TaxonomyToScenarioTask(BaseTask):
    """Task: taxonomy → scenario"""
    
    def get_task_name(self) -> str:
        return "taxonomy_to_scenario"
    
    def load_data(self) -> List[Dict[str, Any]]:
        from src.dataset_generation.utils.data_loading import load_taxonomy_data
        # Use data_path if provided, otherwise get from config.yaml
        data_path = self.args.data_path
        if not data_path:
            from src.dataset_generation import get_default_taxonomy_path
            data_path = get_default_taxonomy_path()
        return load_taxonomy_data(
            data_path, 
            self.args.risk_type, 
            self.args.mechanism
        )
    
    def create_prompter(self):
        return RiskPipelinePrompter(
            self.args.prompt_path, 
            self.task_name, 
            shots_path=self.args.shots_path
        )
    
    def gather_results(self, results_file: str, scenarios_file: str, output_file: str, **kwargs) -> Dict[str, Any]:
        """
        Gather scenarios from generated results and shots.
        Integrated from gather_scenarios.py
        """
        shots_file = kwargs.get('shots_file', None)
        results_dir = kwargs.get('results_dir', None)
        iterate_name = kwargs.get('iterate_name', None)
        source = kwargs.get('source', 'all')
        
        if not shots_file or not results_dir:
            raise ValueError("shots_file and results_dir are required for gather_results")
        
        # Load original shots
        print("Loading original shots...")
        with open(shots_file, 'r', encoding='utf-8') as f:
            shots_data = json.load(f)
        
        # Load generated results
        print("Loading generated results...")
        generated_scenarios = self._load_generated_results(results_dir, iterate_name)
        
        # Build abbreviation maps from original shots
        risk_abbr, mech_abbr, id_width = self._build_abbreviation_maps(shots_data)
        
        # Combine all scenarios with IDs
        all_scenarios = []
        scenario_counter = {}
        
        # Add original shots
        if source in ("all", "original"):
            for risk_type, risk_data in shots_data.items():
                for mechanism, scenarios in risk_data.items():
                    normalized_mechanism = mechanism.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                    key = f"{risk_type}_{normalized_mechanism}"
                    
                    if key not in scenario_counter:
                        scenario_counter[key] = 0
                    
                    for scenario in scenarios:
                        if "id" in scenario and scenario["id"]:
                            scenario_id = scenario["id"]
                        else:
                            scenario_counter[key] += 1
                            scenario_id = self._create_scenario_id(
                                risk_type, mechanism, scenario_counter[key], risk_abbr, mech_abbr, id_width
                            )
                        
                        all_scenarios.append({
                            "id": scenario_id,
                            "risk_type": risk_type,
                            "mechanism": mechanism,
                            "hazard": scenario["hazard"],
                            "action": scenario["action"],
                            "source": "original"
                        })
        
        # Add generated scenarios
        if source in ("all", "generated"):
            for scenario in generated_scenarios:
                normalized_mechanism = scenario['mechanism'].replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                key = f"{scenario['risk_type']}_{normalized_mechanism}"
                
                if key not in scenario_counter:
                    scenario_counter[key] = 0
                
                if "id" in scenario and scenario["id"]:
                    scenario_id = scenario["id"]
                else:
                    scenario_counter[key] += 1
                    scenario_id = self._create_scenario_id(
                        scenario['risk_type'], scenario['mechanism'], scenario_counter[key], risk_abbr, mech_abbr, id_width
                    )
                
                all_scenarios.append({
                    "id": scenario_id,
                    "risk_type": scenario["risk_type"],
                    "mechanism": scenario["mechanism"],
                    "hazard": scenario["hazard"],
                    "action": scenario["action"],
                    "source": scenario["source"]
                })
        
        # Statistics
        total_scenarios = len(all_scenarios)
        original_count = len([s for s in all_scenarios if s['source'] == 'original'])
        generated_count = len([s for s in all_scenarios if s['source'] == 'generated'])
        
        risk_counts = {}
        for scenario in all_scenarios:
            risk_type = scenario['risk_type']
            if risk_type not in risk_counts:
                risk_counts[risk_type] = 0
            risk_counts[risk_type] += 1
        
        result = {
            "metadata": {
                "total_scenarios": total_scenarios,
                "original_count": original_count,
                "generated_count": generated_count,
                "risk_type_counts": risk_counts
            },
            "scenarios": all_scenarios
        }
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save separate files for original and generated
        output_base = os.path.splitext(output_file)[0]
        
        original_scenarios = [s for s in all_scenarios if s['source'] == 'original']
        original_result = {
            "metadata": {
                "total_scenarios": len(original_scenarios),
                "original_count": len(original_scenarios),
                "generated_count": 0,
                "risk_type_counts": {}
            },
            "scenarios": original_scenarios
        }
        for scenario in original_scenarios:
            risk_type = scenario['risk_type']
            if risk_type not in original_result["metadata"]["risk_type_counts"]:
                original_result["metadata"]["risk_type_counts"][risk_type] = 0
            original_result["metadata"]["risk_type_counts"][risk_type] += 1
        
        original_output = f"{output_base}_original.json"
        with open(original_output, 'w', encoding='utf-8') as f:
            json.dump(original_result, f, indent=2, ensure_ascii=False)
        
        generated_scenarios_list = [s for s in all_scenarios if s['source'] == 'generated']
        generated_result = {
            "metadata": {
                "total_scenarios": len(generated_scenarios_list),
                "original_count": 0,
                "generated_count": len(generated_scenarios_list),
                "risk_type_counts": {}
            },
            "scenarios": generated_scenarios_list
        }
        for scenario in generated_scenarios_list:
            risk_type = scenario['risk_type']
            if risk_type not in generated_result["metadata"]["risk_type_counts"]:
                generated_result["metadata"]["risk_type_counts"][risk_type] = 0
            generated_result["metadata"]["risk_type_counts"][risk_type] += 1
        
        generated_output = f"{output_base}_generated.json"
        with open(generated_output, 'w', encoding='utf-8') as f:
            json.dump(generated_result, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        self._save_statistics(result["metadata"], results_dir, iterate_name)
        
        return result
    
    def _load_generated_results(self, results_dir: str, iterate_name: str = None) -> List[Dict[str, Any]]:
        """Load all generated scenario results from results directory"""
        generated_scenarios = []
        
        if iterate_name:
            pattern = os.path.join(results_dir, iterate_name, "taxonomy_to_scenario.json")
            result_files = glob.glob(pattern)
        else:
            patterns = [
                os.path.join(results_dir, "*", "*", "taxonomy_to_scenario_*.json"),
                os.path.join(results_dir, "train_set", "raw", "*", "*", "taxonomy_to_scenario_*.json"),
                os.path.join(results_dir, "*/raw", "*", "*", "taxonomy_to_scenario_*.json")
            ]
            all_files = []
            for pattern in patterns:
                all_files.extend(glob.glob(pattern))
            result_files = list(set([f for f in all_files if "/test/" not in f]))
        
        for file_path in result_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                scenarios_with_metadata = []
                if isinstance(data, dict):
                    if iterate_name:
                        file_risk_type = data.get('type') or data.get('risk_type')
                        file_mechanisms = data.get('mechanism', [])
                        if isinstance(file_mechanisms, list) and len(file_mechanisms) > 0:
                            file_mechanism = file_mechanisms[0].get('name', 'Unknown_Mechanism')
                        elif isinstance(file_mechanisms, dict):
                            file_mechanism = file_mechanisms.get('name', 'Unknown_Mechanism')
                        else:
                            file_mechanism = 'Unknown_Mechanism'
                        metadata = {'risk_type': file_risk_type or 'Unknown_Risk', 'mechanism': file_mechanism}
                    else:
                        path_parts = Path(file_path).parts
                        risk_type = path_parts[-3]
                        mechanism = path_parts[-2]
                        metadata = {'risk_type': risk_type, 'mechanism': mechanism}
                    
                    if 'scenarios' in data:
                        for scenario in data['scenarios']:
                            scenarios_with_metadata.append((scenario, metadata))
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            if iterate_name:
                                file_risk_type = item.get('type') or item.get('risk_type')
                                file_mechanisms = item.get('mechanism', [])
                                if isinstance(file_mechanisms, list) and len(file_mechanisms) > 0:
                                    file_mechanism = file_mechanisms[0].get('name', 'Unknown_Mechanism')
                                else:
                                    file_mechanism = 'Unknown_Mechanism'
                                metadata = {'risk_type': file_risk_type or 'Unknown_Risk', 'mechanism': file_mechanism}
                            else:
                                path_parts = Path(file_path).parts
                                risk_type = path_parts[-3]
                                mechanism = path_parts[-2]
                                metadata = {'risk_type': risk_type, 'mechanism': mechanism}
                            
                            # Handle prediction field (can be dict or list)
                            if 'prediction' in item:
                                prediction = item['prediction']
                                
                                # Case 1: prediction is a dict with 'scenarios'
                                if isinstance(prediction, dict) and 'scenarios' in prediction:
                                    for scenario in prediction['scenarios']:
                                        scenarios_with_metadata.append((scenario, metadata))
                                # Case 2: prediction is a list (old format)
                                elif isinstance(prediction, list):
                                    if len(prediction) > 0 and isinstance(prediction[0], list):
                                        pred_list = prediction[0]
                                        for idx in [0, 1]:
                                            if len(pred_list) > idx and isinstance(pred_list[idx], dict) and 'text' in pred_list[idx]:
                                                try:
                                                    text_content = pred_list[idx]['text']
                                                    parsed_json = json.loads(text_content)
                                                    if 'scenarios' in parsed_json:
                                                        for scenario in parsed_json['scenarios']:
                                                            scenarios_with_metadata.append((scenario, metadata))
                                                        break
                                                except (json.JSONDecodeError, KeyError):
                                                    continue
                            elif 'scenarios' in item:
                                for scenario in item['scenarios']:
                                    scenarios_with_metadata.append((scenario, metadata))
                            elif 'hazard' in item and 'action' in item:
                                scenarios_with_metadata.append((item, metadata))
                
                for scenario, metadata in scenarios_with_metadata:
                    if isinstance(scenario, dict) and 'hazard' in scenario and 'action' in scenario:
                        generated_scenarios.append({
                            'risk_type': metadata['risk_type'],
                            'mechanism': metadata['mechanism'],
                            'hazard': scenario['hazard'],
                            'action': scenario['action'],
                            'source': 'generated'
                        })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        return generated_scenarios
    
    def _abbreviate_name(self, value: str) -> str:
        """Abbreviate a name using the first letter of each word/token."""
        tokens = []
        for part in value.replace("(", " ").replace(")", " ").replace("-", " ").replace("_", " ").split():
            if part:
                tokens.append(part[0].upper())
        return "".join(tokens) or "UNK"
    
    def _build_abbreviation_maps(self, shots_data: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[Tuple[str, str], str], Dict[Tuple[str, str], int]]:
        """Build abbreviation maps from existing shot IDs."""
        risk_abbr = {}
        mech_abbr = {}
        id_width = {}
        
        for risk_type, risk_data in shots_data.items():
            if not isinstance(risk_data, dict):
                continue
            for mechanism, scenarios in risk_data.items():
                if not isinstance(scenarios, list):
                    continue
                for scenario in scenarios:
                    scenario_id = scenario.get("id")
                    if not scenario_id or not isinstance(scenario_id, str):
                        continue
                    parts = scenario_id.split("_")
                    if len(parts) < 3:
                        continue
                    risk_part = parts[0]
                    mech_part = parts[1]
                    index_part = parts[-1]
                    if risk_type not in risk_abbr:
                        risk_abbr[risk_type] = risk_part
                    if (risk_type, mechanism) not in mech_abbr:
                        mech_abbr[(risk_type, mechanism)] = mech_part
                    if (risk_type, mechanism) not in id_width and index_part.isdigit():
                        id_width[(risk_type, mechanism)] = len(index_part)
                    break
        
        return risk_abbr, mech_abbr, id_width
    
    def _create_scenario_id(
        self,
        risk_type: str,
        mechanism: str,
        index: int,
        risk_abbr: Dict[str, str],
        mech_abbr: Dict[Tuple[str, str], str],
        id_width: Dict[Tuple[str, str], int],
    ) -> str:
        """Create unique ID for scenario with abbreviated formatting"""
        risk_part = risk_abbr.get(risk_type) or self._abbreviate_name(risk_type)
        mech_part = mech_abbr.get((risk_type, mechanism)) or self._abbreviate_name(mechanism)
        width = id_width.get((risk_type, mechanism), 2)
        return f"{risk_part}_{mech_part}_{index:0{width}d}"
    
    def _save_statistics(self, statistics: Dict[str, Any], results_dir: str, iterate_name: str = None):
        """Save statistics to scenario_statistics.txt in results directory"""
        if iterate_name:
            output_dir = os.path.join(results_dir, iterate_name)
        else:
            output_dir = results_dir
        
        os.makedirs(output_dir, exist_ok=True)
        stats_file = os.path.join(output_dir, "scenario_statistics.txt")
        
        lines = [
            "=" * 60,
            "Scenario Statistics",
            "=" * 60,
            "",
            f"Total Scenarios: {statistics['total_scenarios']}",
            f"  Original: {statistics['original_count']}",
            f"  Generated: {statistics['generated_count']}",
            "",
            "By Risk Type:",
            "-" * 60,
        ]
        
        for risk_type, count in sorted(statistics['risk_type_counts'].items()):
            lines.append(f"  {risk_type}: {count}")
        
        lines.append("")
        lines.append("=" * 60)
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"\nStatistics saved to: {stats_file}")

