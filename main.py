import sys
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
from dotenv import load_dotenv
import asyncio

load_dotenv()

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from graph.graph import synnoia_agent
from document_schema.schema import Document, DocumentBody
from json_converter.converter_tiptap_to_synnoia import tiptap_to_synnoia
from json_converter.converter_synnoia_to_tiptap import synnoia_to_tiptap

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "synnoia-agent"}


@app.websocket("/ws/agent")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive data from frontend
            data = await websocket.receive_json()
            
            query = data.get("query", "")
            document_name = data.get("document_name", "")
            doc_text = data.get("doc_text", "")
            doc_json = data.get("doc_json", "")
            
            print(f"Received request: query='{query}', doc_json_present={bool(doc_json)}")
            
            # Send acknowledgment
            await websocket.send_json({"status": "processing", "message": "Processing your request..."})
            
            # Prepare initial_state
            initial_state = {
                "query": query,
                "document_name": document_name,
                "doc_text": doc_text,
                "doc_json": doc_json,
                "rephrased_query": "",
                "intent": "",
                "response": "",
                "response_json": Document(document=DocumentBody(nodes=[])),
                "operation_type": "",
                "anchor_id": None
            }
            
            # Convert doc_json to Synnoia format if present
            if doc_json:
                try:
                    print("Converting doc_json to Synnoia format...")
                    # Parse JSON string to dict first
                    doc_json_dict = json.loads(doc_json) if isinstance(doc_json, str) else doc_json
                    synnoia_doc = tiptap_to_synnoia(doc_json_dict)
                    # Convert back to JSON string for the state schema
                    initial_state["doc_json"] = json.dumps(synnoia_doc)
                    print("doc_json conversion complete")
                except Exception as e:
                    print(f"doc_json conversion error: {e}")
                    await websocket.send_json({"error": f"Failed to convert doc_json: {str(e)}"})
                    continue
            
            try:
                print("Invoking agent...")
                
                async def run_agent():
                    return synnoia_agent.invoke(initial_state)
                
                # Run agent with timeout and keep-alive
                result = await asyncio.wait_for(run_agent(), timeout=300)
                print("Agent invocation complete")
            except asyncio.TimeoutError:
                print("Agent execution timeout")
                await websocket.send_json({"error": "Agent execution timeout (5 minutes)"})
                continue
            except Exception as e:
                print(f"Agent execution error: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_json({"error": f"Agent execution failed: {str(e)}"})
                continue
            
            # Prepare response
            response_data = {
                "response": result.get("response", ""),
                "response_json": None,
                "operation_type": result.get("operation_type", ""),
                "anchor_id": result.get("anchor_id", None)
            }
            
            # Convert response_json to TipTap if present
            if result.get("response_json"):
                try:
                    print("Converting response_json to TipTap...")
                    synnoia_data = result["response_json"].model_dump()
                    tiptap_data = synnoia_to_tiptap(synnoia_data)
                    response_data["response_json"] = tiptap_data
                    print("response_json conversion complete")
                except Exception as e:
                    print(f"response_json conversion error: {e}")
                    response_data["response_json"] = {"error": f"Failed to convert response_json: {str(e)}"}
            
            # Send response back to frontend
            print("Sending response to client...")
            print(f"operation_type: {response_data['operation_type']}, anchor_id: {response_data['anchor_id']}")
            await websocket.send_json(response_data)
            print("Response sent successfully")
        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=60,
        ws_ping_interval=15,
        ws_ping_timeout=15
    )



