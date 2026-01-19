# gui/pose_manager.py
"""
Pose 관리 클래스
- 5개 테이블별 포즈 저장/로드
- JSON 파일 관리
"""
import json
from typing import Dict, List, Optional
from config import gui_config


class PoseManager:
    """
    Pose 관리 클래스
    - 테이블별 포즈 저장/로드
    - 파일 입출력
    """
    
    def __init__(self, filepath: str = None):
        """
        Args:
            filepath: JSON 파일 경로 (None이면 기본값)
        """
        self.filepath = filepath or gui_config.POSES_FILE
        self.data = self.load()
    
    def load(self) -> Dict:
        """파일에서 포즈 로드"""
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = None
        except Exception as e:
            print(f"[PoseManager] Load error: {e}")
            data = None
        
        # 기본 구조
        if not data:
            data = self._create_default_data()
        
        # 호환성 체크 및 보완
        data = self._ensure_structure(data)
        
        return data
    
    def save(self) -> bool:
        """파일에 포즈 저장"""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent=2)
            return True
        except Exception as e:
            print(f"[PoseManager] Save error: {e}")
            return False
    
    def _create_default_data(self) -> Dict:
        """기본 데이터 구조 생성"""
        tables = {}
        for i in range(1, 6):
            tables[str(i)] = {
                "pick_pose": gui_config.DEFAULT_PICK_POSE,
                "place_pose": gui_config.DEFAULT_PLACE_POSE,
                "safe_pick": gui_config.DEFAULT_SAFE_PICK,
                "safe_place": gui_config.DEFAULT_SAFE_PLACE
            }
        
        return {
            "tables": tables,
            "num_boxes": gui_config.DEFAULT_NUM_BOXES,
            "box_height": gui_config.DEFAULT_BOX_HEIGHT,
            "approach_offset": gui_config.DEFAULT_APPROACH_OFFSET,
            "vel_normal": 30.0,
            "vel_slow": 30.0
        }
    
    def _ensure_structure(self, data: Dict) -> Dict:
        """데이터 구조 보장 (호환성)"""
        # 기본 구조
        if "tables" not in data:
            data["tables"] = {}
        
        # 5개 테이블 모두 확인
        for i in range(1, 6):
            key = str(i)
            if key not in data["tables"]:
                data["tables"][key] = {
                    "pick_pose": gui_config.DEFAULT_PICK_POSE,
                    "place_pose": gui_config.DEFAULT_PLACE_POSE,
                    "safe_pick": gui_config.DEFAULT_SAFE_PICK,
                    "safe_place": gui_config.DEFAULT_SAFE_PLACE
                }
            else:
                # 4개 포즈 모두 확인
                tbl = data["tables"][key]
                tbl.setdefault("pick_pose", gui_config.DEFAULT_PICK_POSE)
                tbl.setdefault("place_pose", gui_config.DEFAULT_PLACE_POSE)
                tbl.setdefault("safe_pick", gui_config.DEFAULT_SAFE_PICK)
                tbl.setdefault("safe_place", gui_config.DEFAULT_SAFE_PLACE)
        
        # 파라미터 보장
        data.setdefault("num_boxes", gui_config.DEFAULT_NUM_BOXES)
        data.setdefault("box_height", gui_config.DEFAULT_BOX_HEIGHT)
        data.setdefault("approach_offset", gui_config.DEFAULT_APPROACH_OFFSET)
        data.setdefault("vel_normal", 30.0)
        data.setdefault("vel_slow", 30.0)
        
        return data
    
    # -------------------------
    # Getter/Setter
    # -------------------------
    def get_table_poses(self, table: str) -> Dict:
        """특정 테이블의 포즈 가져오기"""
        return self.data["tables"].get(table, {})
    
    def set_table_pose(self, table: str, pose_type: str, pose: List[float]):
        """특정 테이블의 포즈 설정"""
        if table not in self.data["tables"]:
            self.data["tables"][table] = {}
        self.data["tables"][table][pose_type] = pose
    
    def get_param(self, key: str, default=None):
        """파라미터 가져오기"""
        return self.data.get(key, default)
    
    def set_param(self, key: str, value):
        """파라미터 설정"""
        self.data[key] = value