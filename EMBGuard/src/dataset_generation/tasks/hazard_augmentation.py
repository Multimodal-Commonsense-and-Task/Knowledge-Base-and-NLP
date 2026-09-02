"""
Hazard Augmentation Task
Augments scenarios by combining hazards from different scenarios.
"""
from typing import List, Dict, Any, Tuple
import json
import os
import ast
import re
import random
from collections import Counter, defaultdict
from itertools import combinations

from src.dataset_generation.task import BaseTask
from src.dataset_generation.prompters import GraphPostprocessingPrompter


class HazardAugmentationTask(BaseTask):
    """Task: hazard augmentation"""
    
    def get_task_name(self) -> str:
        return "hazard_augmentation"
    
    def load_data(self) -> List[Dict[str, Any]]:
        from src.dataset_generation.utils.data_loading import load_graph_pairs_data
        
        # Determine input scenarios file and pairs file
        iterate_base = self.args.iterate_name.split('/')[0] if self.args.iterate_name else None
        iterate_dir = os.path.join(self.args.save_dir, iterate_base) if iterate_base else None
        
        # Determine input scenarios file (for creating pairs)
        input_scenarios_file = None
        if hasattr(self.args, 'input_scenarios_file') and self.args.input_scenarios_file:
            input_scenarios_file = self.args.input_scenarios_file
            if not os.path.isabs(input_scenarios_file) and iterate_dir:
                input_scenarios_file = os.path.join(iterate_dir, input_scenarios_file)
        elif iterate_dir:
            # Auto-detect: try graphs_normalized.json first, then graphs.json
            potential_file = os.path.join(iterate_dir, "graphs_normalized.json")
            if os.path.exists(potential_file):
                input_scenarios_file = potential_file
            else:
                potential_file = os.path.join(iterate_dir, "graphs.json")
                if os.path.exists(potential_file):
                    input_scenarios_file = potential_file
        
        # Determine pairs file
        pairs_file = None
        if self.args.data_path:
            pairs_file = self.args.data_path
            if not os.path.isabs(pairs_file) and iterate_dir:
                pairs_file = os.path.join(iterate_dir, pairs_file)
        elif iterate_dir:
            pairs_file = os.path.join(iterate_dir, "raw", "hazard_pairs_by_room.json")
        
        # Create pairs if they don't exist
        if not pairs_file or not os.path.exists(pairs_file):
            if not input_scenarios_file or not os.path.exists(input_scenarios_file):
                raise ValueError(
                    "Cannot create pairs: input_scenarios_file is required. "
                    "Please provide --input_scenarios_file or ensure graphs_normalized.json exists in iterate directory."
                )
            
            # Create pairs automatically
            print(f"Pairs file not found. Creating pairs from {input_scenarios_file}...")
            os.makedirs(os.path.dirname(pairs_file), exist_ok=True)
            
            # Get random seed and pairs per mechanism from args or use defaults
            random_seed = getattr(self.args, 'random_seed', 42)
            pairs_per_mechanism = getattr(self.args, 'pairs_per_mechanism', 1)
            self._create_hazard_pairs_by_room(
                input_scenarios_file,
                pairs_file,
                random_seed,
                pairs_per_mechanism
            )
            print(f"✓ Created pairs: {pairs_file}")
        
        # Store pairs_file for gather_results
        self._pairs_file = pairs_file
        self._input_scenarios_file = input_scenarios_file or os.path.join(iterate_dir, "graphs_normalized.json") if iterate_dir else None
        
        return load_graph_pairs_data(pairs_file)
    
    def _pair_identifier(self, pair: Dict[str, Any]) -> Tuple[str, str]:
        """Canonical identifier for a pair so we can dedupe selections."""
        ids = (pair['scenario1_id'], pair['scenario2_id'])
        return tuple(sorted(ids))
    
    def _select_mechanism_coverage_pairs(self, all_pairs: List[Dict[str, Any]], n_per_mechanism: int = 1) -> List[Dict[str, Any]]:
        """Pick n pairs per mechanism combo, covering same and cross combinations."""
        if not all_pairs:
            return []

        def valid_mechanism(value: Any) -> bool:
            if not value:
                return False
            lowered = str(value).strip().lower()
            return lowered != "unknown"

        mechanism_index = defaultdict(list)
        for pair in all_pairs:
            mech1 = pair['scenario1_data'].get('mechanism')
            mech2 = pair['scenario2_data'].get('mechanism')
            if not (valid_mechanism(mech1) and valid_mechanism(mech2)):
                continue

            combo = tuple(sorted((mech1, mech2)))
            mechanism_index[combo].append(pair)

        same_keys = [combo for combo in mechanism_index if combo[0] == combo[1]]
        cross_keys = [combo for combo in mechanism_index if combo[0] != combo[1]]

        random.shuffle(same_keys)
        random.shuffle(cross_keys)

        coverage_pairs = []
        used_identifiers: set[Tuple[str, str]] = set()

        for key_group in (same_keys, cross_keys):
            for combo in key_group:
                pair_list = mechanism_index[combo]
                random.shuffle(pair_list)

                selected_count = 0
                for pair in pair_list:
                    if selected_count >= n_per_mechanism:
                        break
                    
                    pid = self._pair_identifier(pair)
                    if pid in used_identifiers:
                        continue
                    coverage_pairs.append(pair)
                    used_identifiers.add(pid)
                    selected_count += 1

        return coverage_pairs
    
    def _extract_room_from_graph(self, graph: List[List[str]]) -> str:
        """Extract room from graph relationships using existing logic"""
        room_keywords = {
            "kitchen", "bathroom", "bedroom", "living room", "dining room",
            "garage", "basement", "attic", "office", "study", "laundry room",
            "hallway", "entryway", "foyer", "porch", "balcony", "patio",
            "backyard", "yard", "garden", "front porch", "outdoor porch",
            "outdoor area", "staircase", "stairs", "stairwell", "mudroom",
            "pantry", "closet", "playroom", "craft room", "workshop",
            "home office", "study room", "guest room", "child's room",
            "storage room", "utility room", "entry hall", "narrow entryway"
        }
        
        for triplet in graph:
            if len(triplet) >= 3:
                subject, predicate, obj = triplet[0], triplet[1], triplet[2]
                
                if predicate.lower() == "in":
                    obj_lower = obj.lower()
                    
                    if obj_lower in room_keywords:
                        return obj_lower
                    
                    for room_keyword in room_keywords:
                        if room_keyword in obj_lower:
                            return room_keyword
        
        return "unknown"
    
    def _load_scenarios(self, file_path: str) -> List[Dict[str, Any]]:
        """Load scenarios from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both wrapped and unwrapped formats
        if isinstance(data, dict) and 'scenarios' in data:
            return data['scenarios']
        elif isinstance(data, list):
            return data
        else:
            return [data] if data else []
    
    def _group_scenarios_by_room(self, scenarios: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group scenarios by room only"""
        room_groups = defaultdict(list)
        
        for scenario in scenarios:
            graph = scenario.get('graph', [])
            room = self._extract_room_from_graph(graph)
            
            # Add room info to scenario
            scenario_with_room = scenario.copy()
            scenario_with_room['room'] = room
            
            if room == "unknown":
                continue
            room_groups[room].append(scenario_with_room)
        
        return dict(room_groups)
    
    def _create_risk_combination_key(self, risk1: str, risk2: str) -> str:
        """Create a consistent key for risk combinations"""
        # Sort to ensure consistent ordering
        risks = sorted([risk1, risk2])
        return f"{risks[0]}+{risks[1]}"
    
    def _generate_all_possible_pairs(self, room_groups: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Generate all possible pairs within each room"""
        all_pairs = []
        
        for room, scenarios in room_groups.items():
            if len(scenarios) >= 2:
                for s1, s2 in combinations(scenarios, 2):
                    pair = self._create_pair_dict(s1, s2, room)
                    all_pairs.append(pair)
        
        return all_pairs
    
    def _create_pair_dict(self, scenario1: Dict[str, Any], scenario2: Dict[str, Any], room: str) -> Dict[str, Any]:
        """Create a pair dictionary with all necessary metadata"""
        risk1 = scenario1.get('risk_type', '')
        risk2 = scenario2.get('risk_type', '')
        risk_combination = self._create_risk_combination_key(risk1, risk2)
        same_risk_type = (risk1 == risk2)
        
        return {
            "scenario1_id": scenario1['id'],
            "scenario2_id": scenario2['id'],
            "room": room,
            "same_risk_type": same_risk_type,
            "risk_combination": risk_combination,
            # Store full scenario data for downstream processing
            "scenario1_data": {
                "id": scenario1['id'],
                "risk_type": scenario1.get('risk_type', ''),
                "mechanism": scenario1.get('mechanism', ''),
                "hazard": scenario1.get('hazard', ''),
                "action": scenario1.get('action', ''),
                "source": scenario1.get('source', ''),
                "graph": scenario1.get('graph', [])
            },
            "scenario2_data": {
                "id": scenario2['id'],
                "risk_type": scenario2.get('risk_type', ''),
                "mechanism": scenario2.get('mechanism', ''),
                "hazard": scenario2.get('hazard', ''),
                "action": scenario2.get('action', ''),
                "source": scenario2.get('source', ''),
                "graph": scenario2.get('graph', [])
            }
        }
    
    def _sample_pairs_mechanism_coverage(self, all_pairs: List[Dict[str, Any]], n_per_mechanism: int = 1) -> List[Dict[str, Any]]:
        """Sample n pairs per mechanism combination within the same room."""
        coverage_pairs = self._select_mechanism_coverage_pairs(all_pairs, n_per_mechanism)
        random.shuffle(coverage_pairs)
        print(f"Sampled: {len(coverage_pairs)} total mechanism-covering pairs (n={n_per_mechanism} per mechanism)")
        return coverage_pairs
    
    def _create_hazard_pairs_by_room(
        self,
        input_file: str,
        output_file: str,
        seed: int = 42,
        n_per_mechanism: int = 1
    ):
        """Main function to create hazard pairs by room with mechanism coverage sampling"""
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        print(f"Loading scenarios from {input_file}...")
        scenarios = self._load_scenarios(input_file)
        print(f"Loaded {len(scenarios)} scenarios")
        
        print("Grouping scenarios by room...")
        room_groups = self._group_scenarios_by_room(scenarios)

        print("Generating all possible pairs...")
        all_pairs = self._generate_all_possible_pairs(room_groups)
        
        selected_pairs = self._sample_pairs_mechanism_coverage(all_pairs, n_per_mechanism)
        
        # Calculate statistics
        risk_combination_counts = Counter(pair['risk_combination'] for pair in selected_pairs)
        same_risk_count = len([p for p in selected_pairs if p['same_risk_type']])
        different_risk_count = len([p for p in selected_pairs if not p['same_risk_type']])
        room_counts = Counter(pair['room'] for pair in selected_pairs)
        
        # Create output data
        output_data = {
            "metadata": {
                "total_pairs": len(selected_pairs),
                "sampling_strategy": "mechanism_coverage",
                "pairs_per_mechanism": n_per_mechanism,
                "input_file": os.path.basename(input_file),
                "same_risk_pairs": same_risk_count,
                "different_risk_pairs": different_risk_count,
                "risk_combinations": dict(risk_combination_counts),
                "room_distribution": dict(room_counts),
                "random_seed": seed
            },
            "pairs": selected_pairs
        }
        
        # Save output
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nCreated {len(selected_pairs)} pairs:")
        print(f"  - Same risk type: {same_risk_count}")
        print(f"  - Different risk type: {different_risk_count}")
        print(f"  - Rooms covered: {len(room_counts)}")
        
        print(f"\nTop risk combinations:")
        for combination, count in risk_combination_counts.most_common(10):
            print(f"  - {combination}: {count}")
        
        print(f"\nRoom distribution:")
        for room, count in room_counts.most_common():
            print(f"  - {room}: {count}")
        
        print(f"\nOutput saved to: {output_file}")
    
    def create_prompter(self):
        return GraphPostprocessingPrompter(self.args.prompt_path, self.task_name)
    
    def gather_results(self, results_file: str, scenarios_file: str, output_file: str, **kwargs) -> Dict[str, Any]:
        """
        Gather hazard augmented graphs with dual structure.
        Integrated from gather_hazard_augmented_graphs.py
        """
        pairs_file = kwargs.get('pairs_file', None)
        input_scenarios_file = kwargs.get('input_scenarios_file', scenarios_file)
        
        if not pairs_file:
            raise ValueError("pairs_file is required for gather_results")
        
        # Load pairs
        with open(pairs_file, 'r', encoding='utf-8') as f:
            pairs_data = json.load(f)
        pairs = pairs_data.get('pairs', [])
        
        # Load inference results
        inference_mapping = {}
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                results = data
            elif isinstance(data, dict) and 'results' in data:
                results = data['results']
            else:
                results = []
            
            for item in results:
                if isinstance(item, dict):
                    pair_id = item.get('id', '')
                    prediction = item.get('prediction', '')
                    if pair_id:
                        augmented_graph = self._parse_augmented_graph_from_prediction(prediction)
                        inference_mapping[pair_id] = {
                            'augmented_graph': augmented_graph,
                            'prediction': prediction
                        }
        
        # Create dual scenarios
        dual_scenarios = []
        for pair in pairs:
            pair_id_plus = f"{pair['scenario1_id']}+{pair['scenario2_id']}"
            pair_id_dash = f"{pair['scenario1_id']}-{pair['scenario2_id']}"
            
            inference_result = inference_mapping.get(pair_id_plus) or inference_mapping.get(pair_id_dash)
            augmented_graph = inference_result.get('augmented_graph') if inference_result else None
            
            scenario1_data = pair['scenario1_data']
            scenario2_data = pair['scenario2_data']
            
            new_id = f"{scenario1_data['id']}_AUG_{scenario2_data['id']}"
            risk1 = scenario1_data.get('risk_type', '')
            risk2 = scenario2_data.get('risk_type', '')
            mechanism1 = scenario1_data.get('mechanism', '')
            mechanism2 = scenario2_data.get('mechanism', '')
            
            merged_graph = augmented_graph if augmented_graph is not None else scenario1_data.get('graph', [])
            
            dual_scenario = {
                "id": new_id,
                "risk_type": f"{risk1}+{risk2}",
                "mechanism": f"{mechanism1}+{mechanism2}",
                "hazard1": scenario1_data.get('hazard', ''),
                "action1": scenario1_data.get('action', ''),
                "hazard2": scenario2_data.get('hazard', ''),
                "action2": scenario2_data.get('action', ''),
                "source": "hazard_augmented",
                "graph": merged_graph,
                "pair_info": {
                    "scenario1_id": scenario1_data['id'],
                    "scenario2_id": scenario2_data['id'],
                    "room": pair.get('room', ''),
                    "same_risk_type": pair.get('same_risk_type', False)
                }
            }
            dual_scenarios.append(dual_scenario)
        
        result = {
            "metadata": {
                "total_scenarios": len(dual_scenarios),
                "original_count": 0,
                "generated_count": len(dual_scenarios),
                "source": "hazard_augmented"
            },
            "scenarios": dual_scenarios
        }
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Automatically split dual scenarios
        split_output_file = output_file.replace('.json', '_split.json')
        if not split_output_file.endswith('_split.json'):
            split_output_file = output_file.replace('.json', '_split.json')
        
        print(f"\nSplitting dual scenarios...")
        self._split_all_dual_scenarios(
            output_file,
            split_output_file,
            include_dual_metadata=True
        )
        print(f"✓ Split scenarios saved: {split_output_file}")
        
        return result
    
    def _load_dual_scenarios(self, file_path: str) -> Dict[str, Any]:
        """Load dual scenarios from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'scenarios' in data:
                return data
            else:
                raise ValueError(f"Expected dict with 'scenarios' key, got {type(data)}")
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {"metadata": {}, "scenarios": []}
    
    def _split_dual_scenario(self, dual_scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a dual scenario into two independent scenarios"""
        base_id = dual_scenario.get('id', '')
        graph = dual_scenario.get('graph', [])
        pair_info = dual_scenario.get('pair_info', {})
        
        def parse_dual_ids(value: str) -> Tuple[str, str]:
            if "_AUG_" in value:
                parts = value.split("_AUG_", 1)
                if len(parts) == 2 and parts[0] and parts[1]:
                    return parts[0], parts[1]
            if "-" in value:
                parts = value.split("-", 1)
                if len(parts) == 2 and parts[0] and parts[1]:
                    return parts[0], parts[1]
            return pair_info.get('scenario1_id', ''), pair_info.get('scenario2_id', '')
        
        scenario1_id, scenario2_id = parse_dual_ids(base_id)
        
        risk_parts = dual_scenario.get('risk_type', '').split('+')
        mech_parts = dual_scenario.get('mechanism', '').split('+')
        risk1 = risk_parts[0] if risk_parts else ''
        risk2 = risk_parts[1] if len(risk_parts) > 1 else ''
        mech1 = mech_parts[0] if mech_parts else ''
        mech2 = mech_parts[1] if len(mech_parts) > 1 else ''
        
        common_pair_info = {
            "room": pair_info.get('room', ''),
            "same_risk_type": pair_info.get('same_risk_type', False),
            "risk_combination": pair_info.get('risk_combination', ''),
            "input_source": pair_info.get('input_source', '')
        }
        
        # Create scenario 1 (from hazard1/action1)
        scenario1 = {
            "id": scenario1_id,
            "original_dual_id": base_id,
            "risk_type": risk1,
            "mechanism": mech1,
            "hazard": dual_scenario.get('hazard1', ''),
            "action": dual_scenario.get('action1', ''),
            "source": "hazard_augmented_split",
            "graph": graph,
            "pair_info": {
                **common_pair_info,
                "paired_id": scenario2_id,
                "paired_risk_type": risk2,
                "paired_mechanism": mech2,
                "paired_hazard": dual_scenario.get('hazard2', ''),
                "paired_action": dual_scenario.get('action2', '')
            }
        }
        
        # Create scenario 2 (from hazard2/action2)
        scenario2 = {
            "id": scenario2_id,
            "original_dual_id": base_id,
            "risk_type": risk2,
            "mechanism": mech2,
            "hazard": dual_scenario.get('hazard2', ''),
            "action": dual_scenario.get('action2', ''),
            "source": "hazard_augmented_split",
            "graph": graph,
            "pair_info": {
                **common_pair_info,
                "paired_id": scenario1_id,
                "paired_risk_type": risk1,
                "paired_mechanism": mech1,
                "paired_hazard": dual_scenario.get('hazard1', ''),
                "paired_action": dual_scenario.get('action1', '')
            }
        }
        
        # Copy any additional fields (like image_path when available)
        for key, value in dual_scenario.items():
            if key not in ['id', 'risk_type', 'mechanism', 'hazard1', 'action1', 'hazard2', 'action2',
                          'source', 'graph', 'pair_info', 'inference_info']:
                scenario1[key] = value
                scenario2[key] = value
        
        return [scenario1, scenario2]
    
    def _split_all_dual_scenarios(
        self,
        input_file: str,
        output_file: str,
        include_dual_metadata: bool = True
    ):
        """Split all dual scenarios into individual scenarios"""
        
        print(f"Loading dual scenarios from: {input_file}")
        data = self._load_dual_scenarios(input_file)
        
        dual_scenarios = data.get('scenarios', [])
        original_metadata = data.get('metadata', {})
        
        print(f"Found {len(dual_scenarios)} dual scenarios")
        
        # Split all scenarios
        split_scenarios = []
        
        for dual_scenario in dual_scenarios:
            # Check if this is actually a dual scenario
            if 'hazard1' in dual_scenario and 'hazard2' in dual_scenario:
                individual_scenarios = self._split_dual_scenario(dual_scenario)
                split_scenarios.extend(individual_scenarios)
            else:
                # If not a dual scenario, keep as is but mark appropriately
                scenario = dual_scenario.copy()
                scenario['source'] = scenario.get('source', 'original')
                split_scenarios.append(scenario)
        
        # Calculate new metadata
        total_scenarios = len(split_scenarios)
        split_count = len([s for s in split_scenarios if s.get('source') == 'hazard_augmented_split'])
        original_count = total_scenarios - split_count
        
        # Count by risk type
        risk_type_counts = {}
        for scenario in split_scenarios:
            risk_type = scenario.get('risk_type', 'unknown')
            risk_type_counts[risk_type] = risk_type_counts.get(risk_type, 0) + 1
        
        # Count by room (from pair_info)
        room_counts = {}
        for scenario in split_scenarios:
            room = scenario.get('pair_info', {}).get('room', 'unknown')
            room_counts[room] = room_counts.get(room, 0) + 1
        
        # Count same vs different risk types
        same_risk_count = len([s for s in split_scenarios if s.get('pair_info', {}).get('same_risk_type', False)])
        different_risk_count = split_count - same_risk_count
        
        # Create new metadata
        new_metadata = {
            "total_scenarios": total_scenarios,
            "original_count": original_count,
            "split_count": split_count,
            "source": "split_from_dual_scenarios",
            "input_file": os.path.basename(input_file),
            "split_statistics": {
                "dual_scenarios_processed": len(dual_scenarios),
                "individual_scenarios_created": split_count,
                "same_risk_splits": same_risk_count,
                "different_risk_splits": different_risk_count
            },
            "risk_type_counts": risk_type_counts,
            "room_distribution": room_counts
        }
        
        # Include original dual metadata if requested
        if include_dual_metadata:
            new_metadata["original_dual_metadata"] = original_metadata
        
        # Create output data
        output_data = {
            "metadata": new_metadata,
            "scenarios": split_scenarios
        }
        
        # Save output
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n=== Split Summary ===")
        print(f"Dual scenarios processed: {len(dual_scenarios)}")
        print(f"Individual scenarios created: {total_scenarios}")
        print(f"  - From splitting: {split_count}")
        print(f"  - Original scenarios: {original_count}")
        
        if split_count > 0:
            print(f"\nSplit breakdown:")
            print(f"  - Same risk type splits: {same_risk_count}")
            print(f"  - Different risk type splits: {different_risk_count}")
        
        print(f"\nRisk type distribution:")
        for risk_type, count in sorted(risk_type_counts.items()):
            if count > 0:
                print(f"  - {risk_type}: {count}")
        
        print(f"\nOutput saved to: {output_file}")
    
    def _parse_augmented_graph_from_prediction(self, prediction: Any) -> Any:
        """Parse augmented graph from prediction value."""
        try:
            if not isinstance(prediction, str):
                if isinstance(prediction, dict) and 'graph' in prediction:
                    return prediction['graph']
                elif isinstance(prediction, list):
                    return prediction
                return None
            
            cleaned_prediction = prediction.replace('\\n', '\n').strip()
            
            try:
                pred_json = json.loads(cleaned_prediction)
                if isinstance(pred_json, dict) and 'graph' in pred_json:
                    return pred_json['graph']
                elif isinstance(pred_json, list):
                    return pred_json
            except json.JSONDecodeError:
                pass
            
            try:
                pred_literal = ast.literal_eval(cleaned_prediction)
                if isinstance(pred_literal, list):
                    return pred_literal
            except (ValueError, SyntaxError):
                pass
            
            return None
        except Exception:
            return None

