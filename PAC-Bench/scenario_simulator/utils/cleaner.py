from dotenv import load_dotenv
import os

load_dotenv()

class Cleaner:
    def __init__(self, session_id: int):
        if session_id < 0 or session_id >= int(os.getenv("MAX_SESSIONS")):
            raise ValueError(f"session_id must be between 0 and {int(os.getenv('MAX_SESSIONS')) - 1}")
        self.session_id = session_id

    def clean_workspace(self):
        workspace_a = os.getenv("WORKSPACE_AGENT_A").format(session_id=self.session_id)
        workspace_b = os.getenv("WORKSPACE_AGENT_B").format(session_id=self.session_id)
        workspace_shared = os.getenv("WORKSPACE_SHARED").format(session_id=self.session_id)

        for path in [workspace_a, workspace_b, workspace_shared]:
            if os.path.exists(path):
                for root, dirs, files in os.walk(path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
    
    def clean_all(self):
        self.clean_workspace()

if __name__ == "__main__":
    for sid in range(100):
        cleaner = Cleaner(session_id=sid)
        cleaner.clean_all()