from langfuse import get_client
from dotenv import load_dotenv

load_dotenv()

langfuse = get_client()

with langfuse.start_as_current_observation(as_type="span", name="test-connection") as span:
    span.update(output="Langfuse is live")

langfuse.flush()
print("Trace sent successfully")