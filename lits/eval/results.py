from typing import List, Dict, Any, Optional
from ..agents.tree.node import SearchNode
import json
import os
from typing import List, Dict, Union
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)
# -------------------------------------------------------------------
# Assume SearchNode (and its .to_dict()/.from_dict()) is already defined
# and in scope, including the is_continuous attribute.
# -------------------------------------------------------------------

def _slice_dataset(dataset: List[Dict], offset: int, limit: Optional[int]) -> List[Dict]:
    if limit is None:
        return dataset[offset:]
    return dataset[offset : offset + limit]

def prepare_dir(model_name: str, root_dir=None, result_dir_suffix=None, verbose=False):
    # Prepare result directory
    result_dir = os.path.join(
        f"results" if result_dir_suffix is None else f"results_{result_dir_suffix}",
        model_name.split("/")[-1],
    )
    if root_dir is not None:
        result_dir = os.path.join(root_dir, result_dir)
    os.makedirs(result_dir, exist_ok=True)
    
    # Create checkpoints and logs directories
    checkpoints_dir = os.path.join(result_dir, "checkpoints")
    logs_dir = os.path.join(result_dir, "logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    if verbose:
        print(f"Checkpoints will be saved to: {checkpoints_dir}")
        print(f"Logs will be saved to: {logs_dir}")
    return result_dir, checkpoints_dir, logs_dir

class BaseResults(ABC):
    """
    Abstract base for line-oriented result files.
    Subclasses must implement load_results() and append_result().
    """
    def __init__(
        self,
        run_id: str,
        root_dir: Optional[str] = None,
        override: bool = False,
        ext: str = "jsonl"
    ):
        # filepath: root_dir/<classname>_<run_id>.<ext>
        if run_id:
            self.filepath = os.path.join(
                root_dir or ".",
                f"{self.__class__.__name__.lower()}_{run_id}.{ext}"
            )
        else:
            self.filepath = os.path.join(
                root_dir or ".",
                f"{self.__class__.__name__.lower()}.{ext}"
            )

        # If file exists and we're not overriding, load existing results.
        if os.path.isfile(self.filepath):
            if not override:
                print(
                    f"Result file {self.filepath} already exists. "
                    "Loading existing results."
                )
                results = self.load_results(self.filepath)
                # Some subclasses may return (results, extra)
                if isinstance(results, tuple): # e.g., TreeToJsonl
                    self.results, self.ids2nodes = results 
                else:
                    self.results = results
            else:
                os.remove(self.filepath)
                open(self.filepath, "w", encoding="utf-8").close()
                self.results = []
        else:
            # no existing file → start fresh
            open(self.filepath, "w", encoding="utf-8").close()
            self.results = []

        self.loaded = False

    @abstractmethod
    def load_results(self, filepath: str) -> Any:
        ...

    @abstractmethod
    def append_result(self, result: Any) -> None:
        ...
    
    def _append_result(self, label: str):
        self.results.append(label)

class ResultToTxtLine(BaseResults):
    def __init__(self,
        run_id: str,
        root_dir: Optional[str] = None,
        override: bool = False
    ):
        super().__init__(run_id, root_dir, override, ext="txt")

    def load_results(self, filepath: str) -> List[str]:
        with open(filepath, "r", encoding="utf-8") as f:
            preds = [line.strip() for line in f if line.strip()]
        return preds

    def append_result(self, result: str) -> None:
        """
        Append a single label to the text predictions file.
        """
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(result + "\n")
        self._append_result(result)

class TreeToJsonl(BaseResults):
    """
    Utility class to save and load MCTS search trees in JSON-Lines format.
    Each line = one task = a list of root→leaf paths;
    each path = list of SearchNode objects.
    """
    def __init__(
        self,
        run_id: str,
        root_dir: Optional[str] = None,
        override: bool = False,
        node_type: type = SearchNode
    ):
        self.node_type = node_type
        # use .jsonl extension
        super().__init__(run_id, root_dir, override, ext="jsonl")
        

    def load_results(self, filepath: str) -> List[Dict[int, SearchNode]]:
        """
        Reads each line (one task) and reconstructs the full tree:
        - First pass: instantiate each unique node once by its id.
        - Second pass: walk each path to wire up .parent and .children.
        Returns a list of { node_id: node } for each task.
        """
        tasks: List[Dict[int, SearchNode]] = []
        paths_for_all_tasks = []
        self.node_type.reset_id()  # optional: force fresh id counter

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                paths_data = json.loads(line)  
                # paths_data: List[List[Dict]]  (outer list = paths, inner = nodes)

                paths: List[List[SearchNode]] = []
                for path_dicts in paths_data:
                    path: List[SearchNode] = []
                    for nd in path_dicts:
                        path.append(self.node_type.from_dict(nd))
                    paths.append(path)
                paths_for_all_tasks.append(paths)
                    
                # 1) Instantiate each node exactly once
                id2node: Dict[int, SearchNode] = {}
                
                for path_dicts in paths_data:
                    
                    for nd in path_dicts:
                        nid = nd["id"]
                        if nid not in id2node:
                            # restores state, action, children_priority, is_continuous
                            id2node[nid] = self.node_type.from_dict(nd)

                # 2) Link up parent ↔ children pointers
                for path_dicts in paths_data:
                    # walk adjacent node-dict pairs along this path
                    for parent_d, child_d in zip(path_dicts, path_dicts[1:]):
                        parent = id2node[parent_d["id"]]
                        child  = id2node[child_d["id"]]

                        # set back-reference
                        child.parent = parent

                        # append child if not already present
                        if child not in parent.children:
                            parent.children.append(child)

                tasks.append(id2node)

        self.loaded = True
        return paths_for_all_tasks, tasks

    def append_result(self, paths: List[List[SearchNode]]) -> None:
        """
        Append one task to the JSON-Lines file.
        `paths` is a list of root→leaf node lists.
        """
        if len(paths)==0:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps([]) + "\n")
            return
        if paths[0] is None:
            logger.debug("the final trace is None")
            paths = [[SearchNode(state=[], action='')]]
            
        # serialize every node in every path
        serializable: List[List[Dict]] = [
            [node.to_dict() for node in path]
            for path in paths
        ]

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(serializable) + "\n")

        # keep in-memory copy
        self._append_result(paths)
        
    def get_all_paths_with_node_ids(self, example_idx: int, verbose=False) -> List[List[SearchNode]]:
        final_trace_plus_all = self.results[example_idx] 
        paths_with_node_ids = []
        for i, path in enumerate(final_trace_plus_all):
            nds = [node.id for node in path]
            if verbose:
                if i == 0:
                    print(f"Solution Path: {nds}")
                else:
                    print(f"Path in One Iteration: {nds}")
            paths_with_node_ids.append(nds)
        return paths_with_node_ids


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append( json.loads(line))
            except json.JSONDecodeError:
                print(f"============= Error decoding JSON: {line} =============")
                continue
    return lines

def parse_reasoning_and_label(text: str, think_prefix="<think>", think_suffix="</think>", extract_label: str="after_think", truth: str = None) -> Dict[str, Any]:
    """
    Extract reasoning enclosed in <think>...</think> and the final label from the given text.
    If tags are missing or parsing fails, return {'reasoning': None, 'label': None, 'text': text}.
    """
    # Only parse if both tags are present
    if think_prefix in text and think_suffix in text:
        try:
            start = text.index(think_prefix) + len(think_prefix)
            end = text.index(think_suffix)
            reasoning = text[start:end].strip()
            if extract_label == "last_line":
                # Label is on the last non-empty line
                label = text.strip().splitlines()[-1].strip()
            elif extract_label == "after_think":
                # Label is after the </think> tag
                label = text[end+len(think_suffix):].strip()
                # remove ".", "\n" if label begins/ends with them
                if label.startswith(".") or label.startswith("\n"):
                    label = label[1:]
                if label.endswith(".") or label.endswith("\n"):
                    label = label[:-1]
            result = {"reasoning": reasoning, "label": label}
        except (ValueError, IndexError):
            # Fall through to default
            result = {"reasoning": None, "label": None, "text": text}
    else:
        # Fallback when parsing unsuccessful
        result = {"reasoning": None, "label": None, "text": text}
    if truth is not None:
        result["truth"] = truth
    return result

class ResultDictToJsonl(BaseResults):
    def __init__(
        self,
        run_id: str,
        root_dir: Optional[str] = None,
        override: bool = False
    ):
        # use .jsonl extension
        super().__init__(run_id, root_dir, override, ext="jsonl")
    
    def load_results(self, filepath: str) -> List[Dict[str, Any]]:
        preds = load_jsonl(filepath)
        return preds

    def append_result(self, result: Union[str, Dict[str, Any]], truth: str = None) -> None:
        """
        Append a structured JSON entry.
        """
        if isinstance(result, str):
            entry = parse_reasoning_and_label(result, truth=truth)
        elif isinstance(result, dict):
            entry = result
        else:
            raise ValueError(f"Unsupported result type: {type(result)}")
        
        with open(self.filepath, "a", encoding="utf-8") as f:
            json.dump(entry, f)
            f.write("\n")
        self._append_result(entry)


class ResultDictToCSV(BaseResults):
    """
    Utility class to save evaluation results to CSV format with immediate write-through.
    
    Each result is a dictionary that gets appended as a new row in the CSV file.
    The CSV header is written on first append based on the keys in the first result.
    
    Usage:
        result_saver = ResultDictToCSV(
            run_id='eval',
            root_dir='results/model_name',
            override=False,
            exclude_columns=['trajectory']  # Optional: columns to exclude from CSV
        )
        
        # Append results one at a time (immediately written to file)
        result_saver.append_result({
            'idx': 0,
            'answer': 'yes',
            'num_calls': 10,
            'input_tokens': 1000,
            'output_tokens': 500
        })
    """
    def __init__(
        self,
        run_id: str,
        root_dir: Optional[str] = None,
        override: bool = False,
        exclude_columns: Optional[List[str]] = None
    ):
        """
        Args:
            run_id: Identifier for this result file
            root_dir: Directory to save the CSV file
            override: If True, overwrite existing file; if False, append to it
            exclude_columns: List of column names to exclude from CSV output
        """
        self.exclude_columns = exclude_columns or []
        self.header_written = False
        self.columns = None
        
        # use .csv extension
        super().__init__(run_id, root_dir, override, ext="csv")
        
        # Check if file exists and has content (header already written)
        if os.path.isfile(self.filepath) and os.path.getsize(self.filepath) > 0:
            self.header_written = True
            # Read existing columns from header using pandas
            import pandas as pd
            try:
                df = pd.read_csv(self.filepath, nrows=0)
                self.columns = list(df.columns)
            except Exception:
                self.columns = None
    
    def load_results(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load existing CSV results into a list of dictionaries using pandas.
        """
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
            return df.to_dict('records')
        except FileNotFoundError:
            return []
        except Exception as e:
            logger.warning(f"Error loading CSV: {e}")
            return []

    def append_result(self, result: Dict[str, Any]) -> None:
        """
        Append a single result dictionary as a new row in the CSV file using pandas.
        
        The result is immediately written to disk (no buffering).
        On first call, the CSV header is written based on the result keys.
        
        Args:
            result: Dictionary containing the evaluation result
        """
        import pandas as pd
        
        if not isinstance(result, dict):
            raise ValueError(f"Result must be a dictionary, got {type(result)}")
        
        # Filter out excluded columns
        filtered_result = {
            k: v for k, v in result.items() 
            if k not in self.exclude_columns
        }
        
        # Determine columns from first result
        if not self.header_written:
            self.columns = list(filtered_result.keys())
            
            # Write header with pandas
            df = pd.DataFrame([filtered_result])
            df.to_csv(self.filepath, index=False, encoding='utf-8')
            
            self.header_written = True
        else:
            # Append the row using pandas
            # Ensure all columns are present
            row_to_write = {k: filtered_result.get(k, '') for k in self.columns}
            df = pd.DataFrame([row_to_write])
            df.to_csv(self.filepath, mode='a', header=False, index=False, encoding='utf-8')
        
        # Keep in-memory copy
        self._append_result(filtered_result)
    
    def update_column(self, row_identifier: str, identifier_value: Any, 
                     column_updates: Dict[str, Any], create_if_missing: bool = True) -> None:
        """
        Update specific columns for rows matching the identifier without re-evaluating.
        
        This allows adding new evaluation perspectives (columns) to existing results
        without re-running expensive LLM evaluations.
        
        Args:
            row_identifier: Column name to identify rows (e.g., 'idx')
            identifier_value: Value to match in the identifier column
            column_updates: Dictionary of {column_name: new_value} to update
            create_if_missing: If True, add new columns to CSV if they don't exist
        
        Example:
            # Add a new evaluation perspective without re-running previous evaluations
            saver = ResultDictToCSV(run_id='eval', root_dir='results')
            
            # Update row where idx=5 with new evaluation perspective
            saver.update_column(
                row_identifier='idx',
                identifier_value=5,
                column_updates={'new_perspective': 'yes', 'new_score': 0.95}
            )
        """
        import pandas as pd
        
        # Load existing results as DataFrame
        try:
            df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            logger.warning(f"No existing results found in {self.filepath}")
            return
        
        if df.empty:
            logger.warning(f"CSV file is empty: {self.filepath}")
            return
        
        # Find rows to update
        mask = df[row_identifier].astype(str) == str(identifier_value)
        
        if not mask.any():
            logger.warning(
                f"No rows found with {row_identifier}={identifier_value}"
            )
            return
        
        # Update columns for matching rows
        for col, val in column_updates.items():
            if col not in self.exclude_columns:
                df.loc[mask, col] = val
        
        # Preserve original column order and add new columns at the end
        if self.columns:
            existing_cols = [c for c in self.columns if c in df.columns]
            new_cols = [c for c in df.columns if c not in self.columns]
            ordered_columns = existing_cols + sorted(new_cols)
            df = df[ordered_columns]
        
        # Write updated DataFrame back to CSV
        df.to_csv(self.filepath, index=False, encoding='utf-8')
        
        # Update internal state
        self.columns = list(df.columns)
        self.results = df.to_dict('records')
        
        logger.info(
            f"Updated {row_identifier}={identifier_value} with columns: "
            f"{list(column_updates.keys())}"
        )
    
    def update_columns_batch(self, row_identifier: str, 
                            updates: List[Dict[str, Any]]) -> None:
        """
        Batch update multiple rows with new column values using pandas.
        
        More efficient than calling update_column() multiple times as it only
        rewrites the CSV file once.
        
        Args:
            row_identifier: Column name to identify rows (e.g., 'idx')
            updates: List of dicts, each containing:
                - identifier_value: Value to match in identifier column
                - column_updates: Dict of {column_name: new_value}
        
        Example:
            # Add new evaluation perspective to multiple rows at once
            saver = ResultDictToCSV(run_id='eval', root_dir='results')
            
            updates = [
                {'identifier_value': 0, 'column_updates': {'new_eval': 'yes'}},
                {'identifier_value': 1, 'column_updates': {'new_eval': 'no'}},
                {'identifier_value': 2, 'column_updates': {'new_eval': 'yes'}},
            ]
            saver.update_columns_batch(row_identifier='idx', updates=updates)
        """
        import pandas as pd
        
        # Load existing results as DataFrame
        try:
            df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            logger.warning(f"No existing results found in {self.filepath}")
            return
        
        if df.empty:
            logger.warning(f"CSV file is empty: {self.filepath}")
            return
        
        # Apply updates
        updated_count = 0
        for update in updates:
            identifier_val = str(update['identifier_value'])
            column_updates = update['column_updates']
            
            # Find matching rows
            mask = df[row_identifier].astype(str) == identifier_val
            
            if mask.any():
                # Update columns for matching rows
                for col, val in column_updates.items():
                    if col not in self.exclude_columns:
                        df.loc[mask, col] = val
                updated_count += mask.sum()
        
        if updated_count == 0:
            logger.warning("No rows were updated")
            return
        
        # Preserve original column order and add new columns at the end
        if self.columns:
            existing_cols = [c for c in self.columns if c in df.columns]
            new_cols = [c for c in df.columns if c not in self.columns]
            ordered_columns = existing_cols + sorted(new_cols)
            df = df[ordered_columns]
        
        # Write updated DataFrame back to CSV
        df.to_csv(self.filepath, index=False, encoding='utf-8')
        
        # Update internal state
        self.columns = list(df.columns)
        self.results = df.to_dict('records')
        
        logger.info(f"Batch updated {updated_count} rows")

