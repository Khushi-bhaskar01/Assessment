import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserIdRequest(BaseModel):
    user_id: str

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    user_id: str
    generated_goal: Optional[str] = None
    roadmap: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None

class AssessmentResponse(BaseModel):
    success: bool
    message: str
    user_id: str
    assessments_generated: Optional[int] = None
    assessments_kept: Optional[int] = None
    assessments_removed: Optional[int] = None
    total_assessments: Optional[int] = None
    error_details: Optional[str] = None

# FIXED Firebase initialization function
def initialize_firebase():
    """Initialize Firebase Admin SDK with multiple fallback methods"""
    try:
        # Check if Firebase is already initialized
        try:
            firebase_admin.get_app()
            print("Firebase already initialized")
            return True
        except ValueError:
            pass
        
        print("Starting Firebase initialization...")
        
        # Method 1: Environment variable with JSON string (for production)
        service_account_key = os.environ.get('GOOGLE_SERVICE_ACCOUNT_KEY')
        if service_account_key:
            try:
                print("Attempting Firebase initialization with environment variable...")
                service_account_info = json.loads(service_account_key)
                cred = credentials.Certificate(service_account_info)
                firebase_admin.initialize_app(cred)
                print("✅ Firebase initialized successfully using environment variable")
                return True
            except Exception as e:
                print(f"❌ Failed with environment variable: {e}")
        else:
            print("No GOOGLE_SERVICE_ACCOUNT_KEY found in environment")
        
        # Method 2: Base64 encoded credentials (alternative for production)
        encoded_key = os.environ.get('GOOGLE_SERVICE_ACCOUNT_BASE64')
        if encoded_key:
            try:
                print("Attempting Firebase initialization with base64 credentials...")
                import base64
                decoded_key = base64.b64decode(encoded_key).decode('utf-8')
                service_account_info = json.loads(decoded_key)
                cred = credentials.Certificate(service_account_info)
                firebase_admin.initialize_app(cred)
                print("✅ Firebase initialized successfully using base64 credentials")
                return True
            except Exception as e:
                print(f"❌ Failed with base64 credentials: {e}")
        else:
            print("No GOOGLE_SERVICE_ACCOUNT_BASE64 found in environment")
        
        # Method 3: File-based (for local development) - Fixed path handling
        credential_paths = [
            "pacific-vault-470814-s5-25cf6513ecef.json",  # Root directory
            "modell/pacific-vault-470814-s5-25cf6513ecef.json",  # Unix-style path
            os.path.join("modell", "pacific-vault-470814-s5-25cf6513ecef.json"),  # Cross-platform
        ]
        
        for path in credential_paths:
            if os.path.exists(path):
                try:
                    print(f"Attempting Firebase initialization with file: {path}")
                    cred = credentials.Certificate(path)
                    firebase_admin.initialize_app(cred)
                    print(f"✅ Firebase initialized successfully using file: {path}")
                    return True
                except Exception as e:
                    print(f"❌ Failed with {path}: {e}")
        
        # Method 4: Application Default Credentials (if running on Google Cloud)
        try:
            print("Attempting Firebase initialization with default credentials...")
            firebase_admin.initialize_app()
            print("✅ Firebase initialized successfully using default credentials")
            return True
        except Exception as e:
            print(f"❌ Failed with default credentials: {e}")
        
        print("❌ All Firebase initialization methods failed")
        return False
        
    except Exception as e:
        print(f"❌ Error in Firebase initialization: {e}")
        return False

# Initialize Firebase when the module loads
firebase_initialized = initialize_firebase()
if firebase_initialized:
    db = firestore.client()
    print("✅ Firestore client created successfully")
else:
    db = None
    print("❌ Firestore client not created - Firebase initialization failed")

@app.get("/debug-firebase")
async def debug_firebase():
    """Debug Firebase initialization issues"""
    try:
        # Check environment variables
        json_key = os.environ.get('GOOGLE_SERVICE_ACCOUNT_KEY')
        base64_key = os.environ.get('GOOGLE_SERVICE_ACCOUNT_BASE64')
        google_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        
        debug_info = {
            "firebase_initialized": firebase_initialized,
            "db_object_exists": db is not None,
            "environment_variables": {
                "GOOGLE_SERVICE_ACCOUNT_KEY_exists": bool(json_key),
                "GOOGLE_SERVICE_ACCOUNT_KEY_length": len(json_key) if json_key else 0,
                "GOOGLE_SERVICE_ACCOUNT_BASE64_exists": bool(base64_key),
                "GOOGLE_APPLICATION_CREDENTIALS": google_creds,
            },
            "current_directory": os.getcwd(),
            "files_in_directory": os.listdir('.') if os.path.exists('.') else [],
        }
        
        # Check if modell directory exists
        if os.path.exists('modell'):
            debug_info["modell_directory_contents"] = os.listdir('modell')
        
        # Try to manually test Firebase initialization
        test_result = "not_attempted"
        try:
            if json_key:
                test_json = json.loads(json_key)
                test_result = f"JSON parse successful - project_id: {test_json.get('project_id', 'missing')}"
            else:
                test_result = "No JSON key found in environment"
        except Exception as e:
            test_result = f"JSON parse failed: {str(e)}"
        
        debug_info["manual_test"] = test_result
        
        # Test database connection if Firebase is initialized
        if firebase_initialized and db:
            try:
                # Try a simple query
                test_ref = db.collection('user').limit(1)
                docs = list(test_ref.get())
                debug_info["database_test"] = f"✅ Database connection successful - found {len(docs)} documents"
            except Exception as e:
                debug_info["database_test"] = f"❌ Database connection failed: {str(e)}"
        else:
            debug_info["database_test"] = "❌ Cannot test - Firebase not initialized"
        
        return debug_info
        
    except Exception as e:
        return {
            "error": str(e),
            "firebase_initialized": firebase_initialized
        }

def get_user_skills(user_id: str) -> Dict[str, Any]:
    """Retrieves the skills map from a user document."""
    if not db:
        raise Exception("Firebase not initialized")
    
    try:
        user_doc_ref = db.collection('user').document(user_id)
        user_doc_snapshot = user_doc_ref.get()

        if user_doc_snapshot.exists:
            user_data = user_doc_snapshot.to_dict()
            print("Retrieved user data:", user_data)
            
            # Check for both "skills" and "skills " to handle data inconsistency
            if 'skills' in user_data:
                skills_data = user_data['skills']
            elif 'skills ' in user_data:
                skills_data = user_data['skills ']
            else:
                skills_data = {}

            return skills_data
        else:
            raise Exception(f"No user document found with ID '{user_id}'")

    except Exception as e:
        raise Exception(f"Error retrieving user skills: {str(e)}")

def goal_generator(skills_data: dict) -> str:
    """Generates a career goal based on user's skills."""
    
    if not skills_data:
        raise Exception("Skills data is empty or invalid")

    try:
        llm = ChatVertexAI(
            model="gemini-2.0-flash-exp", 
            temperature=0.3, 
            max_output_tokens=256, 
            project="pacific-vault-470814-s5"
        )

        # Convert the skills dictionary into a human-readable string for the LLM
        primary_skills = skills_data.get('primary', [])
        secondary_skills = skills_data.get('secondary', [])
        
        formatted_skills = f"Primary Skills: {', '.join(primary_skills)}. Secondary Skills: {', '.join(secondary_skills)}."

        prompt = ChatPromptTemplate([
            ("system", 
            "You are an expert career advisor. Based on the user's skills, suggest a specific and realistic career goal. "
            "The goal should be achievable within 1-2 years and should leverage their existing skills while helping them grow. "
            "Provide only the career goal as a concise statement (e.g., 'Full Stack Web Developer', 'Data Analyst specializing in Business Intelligence', etc.). "
            "Do not include explanations, just the career goal title."),
            ("user", "Based on my skills: {skills}, what career goal should I pursue?"),
        ])

        chain = prompt | llm
        response = chain.invoke({"skills": formatted_skills})
        career_goal = response.content.strip().strip('"\'')
        print(f"Generated Career Goal: {career_goal}")
        return career_goal
    except Exception as e:
        raise Exception(f"Failed to generate career goal: {str(e)}")

def roadmap_generator(skills_data: dict, user_goal: str) -> dict:
    """Generates a career or skill roadmap in dictionary format based on user background and goals."""
    
    if not skills_data or not user_goal:
        raise Exception("Skills data or user goal is empty or invalid")

    try:
        llm = ChatVertexAI(
            model="gemini-2.0-flash-exp", 
            temperature=0.2, 
            max_output_tokens=1024, 
            project="pacific-vault-470814-s5"
        )

        # Convert the skills dictionary into a human-readable string for the LLM
        primary_skills = skills_data.get('primary', [])
        secondary_skills = skills_data.get('secondary', [])
        
        formatted_skills = f"Primary Skills: {', '.join(primary_skills)}. Secondary Skills: {', '.join(secondary_skills)}."

        prompt = ChatPromptTemplate([
            ("system", 
            "You are an expert career and skill roadmap generator. "
            "Generate a concise roadmap with 3-5 key tasks based on the user's skills and their goal. "
            "Return ONLY a valid JSON object with NO additional text, explanations, or markdown formatting. "
            "Use this exact structure: "
            '{{"title": "string describing the roadmap", '
            '"tasks": [{{"status": "completed/current/upcoming", "taskName": "specific task name", "timeAllocation": "time estimate", "resources": [{{"title": "resource name", "link": "https://valid-url.com"}}]}}]}}. '
            'Ensure all JSON values are properly quoted strings. Make sure the JSON is complete and valid.'),
            ("user", "My skills: {skills}. Generate a roadmap to become: {goal}"),
        ])

        chain = prompt | llm
        response = chain.invoke({"skills": formatted_skills, "goal": user_goal})
        
        # More robust JSON extraction
        raw_content = response.content.strip()
        print(f"Raw LLM Response: {raw_content}")
        
        # Remove any markdown code blocks or extra formatting
        raw_content = raw_content.replace('```json', '').replace('```', '').strip()
        
        # Find the start and end of the JSON object
        json_start = raw_content.find('{')
        json_end = raw_content.rfind('}')
        
        if json_start != -1 and json_end != -1:
            clean_json_str = raw_content[json_start:json_end + 1]
        else:
            raise Exception("No valid JSON found in LLM response")
        
        print(f"Extracted JSON: {clean_json_str}")
        
        # Parse the JSON
        roadmap = json.loads(clean_json_str)
        
        # Validate the roadmap structure
        if not isinstance(roadmap, dict):
            raise Exception("Roadmap is not a dictionary")
        
        if "title" not in roadmap or "tasks" not in roadmap:
            raise Exception("Roadmap missing required fields (title, tasks)")
        
        if not isinstance(roadmap.get("tasks"), list):
            raise Exception("Tasks must be a list")
        
        if len(roadmap.get("tasks", [])) == 0:
            raise Exception("Roadmap must contain at least one task")
        
        print("Roadmap validation successful")
        return roadmap
        
    except json.JSONDecodeError as e:
        raise Exception(f"JSON parsing error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error in roadmap generation: {str(e)}")

def update_user_data(user_id: str, goal: str, roadmap_data: dict) -> bool:
    """Updates the user's document in Firestore with the generated goal and roadmap."""
    if not db:
        raise Exception("Firebase not initialized")
    
    try:
        user_doc_ref = db.collection('user').document(user_id)
        
        # Validate roadmap_data before proceeding
        if not roadmap_data:
            raise Exception("Invalid roadmap data provided")
        
        # Get existing document
        existing_doc = user_doc_ref.get()
        
        if existing_doc.exists:
            user_data = existing_doc.to_dict()
            existing_roadmaps = user_data.get('roadmap', [])
            
            # Ensure existing_roadmaps is a list and filter out any invalid entries
            if not isinstance(existing_roadmaps, list):
                existing_roadmaps = []
            else:
                # Remove any blank or invalid roadmaps
                existing_roadmaps = [
                    rm for rm in existing_roadmaps 
                    if rm and isinstance(rm, dict) and rm.get('title') and rm.get('tasks')
                ]
        else:
            existing_roadmaps = []
        
        # Add new roadmap to the array
        existing_roadmaps.append(roadmap_data)
        
        # Update the document
        user_doc_ref.update({
            'generated_goal': goal,
            'roadmap': existing_roadmaps
        })
        
        print(f"Successfully updated user document '{user_id}' with the generated goal and roadmap.")
        print(f"Total valid roadmaps for user: {len(existing_roadmaps)}")
        return True
        
    except Exception as e:
        raise Exception(f"Error updating Firestore: {str(e)}")

class AssessmentService:
    """Assessment service for generating and managing skill assessments"""
    
    def __init__(self, db_client, vertex_ai_project):
        """Initialize with existing database client"""
        self.db = db_client
        self.vertex_ai_project = vertex_ai_project
    
    def get_user_skills(self, user_id: str):
        """Retrieve user skills arrays (primary and secondary) by user_id"""
        try:
            user_ref = self.db.collection('user').document(user_id)
            user_doc = user_ref.get()
            
            if user_doc.exists:
                user_data = user_doc.to_dict()
                skill_map = user_data.get('skills', {})
                primary_skills = skill_map.get('primary', [])
                secondary_skills = skill_map.get('secondary', [])
                return primary_skills, secondary_skills
            else:
                raise Exception(f"User with ID '{user_id}' not found")
                
        except Exception as e:
            raise Exception(f"Error retrieving user skills: {str(e)}")
    
    def assessment_generator(self, skill: str, category: str) -> dict:
        """Generates an assessment in JSON format for a given skill + category"""
        try:
            llm = ChatVertexAI(
                model="gemini-2.0-flash-exp",
                temperature=0.3,
                max_output_tokens=2048,
                project=self.vertex_ai_project
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are an assessment generator that creates technical skill assessments. "
                 "Generate exactly 10 multiple-choice questions for the given skill. "
                 "Return a valid JSON object with this exact structure:\n"
                 '{{\n'
                 '  "skill": string,\n'
                 '  "category": string,\n'
                 '  "questions": [\n'
                 '    {{\n'
                 '      "questionId": string (format: "q1", "q2", etc.),\n'
                 '      "questionText": string,\n'
                 '      "options": [string, string, string, string] (exactly 4 options),\n'
                 '      "correctOption": string (must match one of the options exactly)\n'
                 '    }}\n'
                 '  ]\n'
                 '}}\n\n'
                 "Important rules:\n"
                 "- Generate exactly 10 questions\n"
                 "- Each question must have exactly 4 options (A, B, C, D format)\n"
                 "- correctOption must be the full text of the correct answer, not just A/B/C/D\n"
                 "- Questions should test practical knowledge and real-world application\n"
                 "- Make questions progressively challenging (easy to advanced)\n"
                 "- Ensure JSON is valid and properly formatted"),
                ("user", "Generate a 10-question assessment for skill: {skill}, category: {category}. "
                         "Focus on practical, industry-relevant questions that test real understanding.")
            ])

            chain = prompt | llm
            response = chain.invoke({"skill": skill, "category": category})

            raw_content = response.content.strip()
            
            # Extract JSON safely
            json_start = raw_content.find('{')
            json_end = raw_content.rfind('}')
            clean_json_str = raw_content[json_start:json_end+1] if json_start != -1 else raw_content

            assessment = json.loads(clean_json_str)
            
            # Validate the assessment structure
            if self.validate_assessment_structure(assessment, skill, category):
                return assessment
            else:
                raise Exception("Assessment structure validation failed")
                
        except json.JSONDecodeError as e:
            raise Exception(f"JSON parsing error: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to generate assessment for {skill}: {str(e)}")
    
    def validate_assessment_structure(self, assessment, expected_skill, expected_category):
        """Validate that the generated assessment matches the expected structure"""
        try:
            # Check basic structure
            if not isinstance(assessment, dict):
                return False
            
            # Check required fields
            if "questions" not in assessment:
                return False
            
            questions = assessment["questions"]
            if not isinstance(questions, list) or len(questions) != 10:
                return False
            
            # Validate each question
            for question in questions:
                if not isinstance(question, dict):
                    return False
                
                required_fields = ["questionId", "questionText", "options", "correctOption"]
                for field in required_fields:
                    if field not in question:
                        return False
                
                # Validate options
                options = question["options"]
                if not isinstance(options, list) or len(options) != 4:
                    return False
                
                # Validate correctOption is one of the options
                if question["correctOption"] not in options:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_existing_assessments(self, user_id: str):
        """Get existing assessments for a user"""
        try:
            user_ref = self.db.collection('user').document(user_id)
            user_doc = user_ref.get()
            
            if user_doc.exists:
                user_data = user_doc.to_dict()
                return user_data.get('assessment', [])
            else:
                return []
                
        except Exception as e:
            raise Exception(f"Error retrieving existing assessments: {str(e)}")
    
    def create_skill_key(self, skill: str, category: str):
        """Create a unique key for skill + category combination"""
        return f"{skill.strip().lower()}_{category.strip().lower()}"
    
    def initialize_user_assessment_array(self, user_id: str):
        """Initialize the assessment field as an empty array if it doesn't exist"""
        try:
            user_ref = self.db.collection('user').document(user_id)
            user_doc = user_ref.get()
            
            if user_doc.exists:
                user_data = user_doc.to_dict()
                if 'assessment' not in user_data or not isinstance(user_data.get('assessment'), list):
                    user_ref.update({"assessment": []})
                    print(f"Initialized assessment array for user {user_id}")
            else:
                raise Exception(f"User document {user_id} does not exist")
                
        except Exception as e:
            raise Exception(f"Error initializing assessment array: {str(e)}")
    
    def sync_assessments_with_skills(self, user_id: str):
        """Sync assessments with current skills"""
        print(f"\nSyncing assessments for user: {user_id}")
        
        # Initialize assessment array
        self.initialize_user_assessment_array(user_id)
        
        # Get current user skills
        primary_skills, secondary_skills = self.get_user_skills(user_id)
        
        print(f"Primary skills ({len(primary_skills)}): {primary_skills}")
        print(f"Secondary skills ({len(secondary_skills)}): {secondary_skills}")
        
        # Create set of current skills with their categories
        current_skill_keys = set()
        current_skills_map = {}
        
        for skill in primary_skills:
            key = self.create_skill_key(skill, "primary")
            current_skill_keys.add(key)
            current_skills_map[key] = {"skill": skill, "category": "primary"}
        
        for skill in secondary_skills:
            key = self.create_skill_key(skill, "secondary")
            current_skill_keys.add(key)
            current_skills_map[key] = {"skill": skill, "category": "secondary"}
        
        # Get existing assessments
        existing_assessments = self.get_existing_assessments(user_id)
        print(f"Found {len(existing_assessments)} existing assessments")
        
        # Create set of existing assessment keys
        existing_skill_keys = set()
        assessments_to_keep = []
        
        for assessment in existing_assessments:
            existing_skill = assessment.get('skill', '').strip()
            existing_category = assessment.get('category', '').strip()
            key = self.create_skill_key(existing_skill, existing_category)
            existing_skill_keys.add(key)
            
            # Keep assessment if skill still exists
            if key in current_skill_keys:
                assessments_to_keep.append(assessment)
                print(f"Keeping assessment for {existing_skill} ({existing_category})")
            else:
                print(f"Removing assessment for {existing_skill} ({existing_category})")
        
        # Find skills that need new assessments
        skills_needing_assessments = current_skill_keys - existing_skill_keys
        
        print(f"Assessments to keep: {len(assessments_to_keep)}")
        print(f"Skills needing new assessments: {len(skills_needing_assessments)}")
        
        # Generate assessments for missing skills
        successful_generations = 0
        
        if skills_needing_assessments:
            print("\nGenerating missing assessments...")
            
            for skill_key in skills_needing_assessments:
                skill_info = current_skills_map[skill_key]
                skill_name = skill_info["skill"]
                category = skill_info["category"]
                
                print(f"Generating assessment for {skill_name} ({category})")
                
                try:
                    assessment = self.assessment_generator(skill_name, category)
                    
                    # Create new assessment object
                    questions = assessment.get('questions', [])
                    new_assessment_obj = {
                        "skill": skill_name,
                        "category": category,
                        "status": "not_started",
                        "score": "",
                        "totalQuestions": len(questions),
                        "attemptedQuestions": "",
                        "correctAnswers": "",
                        "questions": [],
                        "attemptedAt": "",
                        "completedAt": ""
                    }
                    
                    # Process questions to match schema
                    for question in questions:
                        question_obj = {
                            "questionId": question.get("questionId", ""),
                            "questionText": question.get("questionText", ""),
                            "options": question.get("options", []),
                            "correctOption": question.get("correctOption", ""),
                            "userAnswer": ""
                        }
                        new_assessment_obj["questions"].append(question_obj)
                    
                    assessments_to_keep.append(new_assessment_obj)
                    successful_generations += 1
                    print(f"Successfully generated assessment for {skill_name}")
                    
                except Exception as e:
                    print(f"Failed to generate assessment for {skill_name}: {str(e)}")
        
        # Update the user document with the synchronized assessments
        user_ref = self.db.collection('user').document(user_id)
        user_ref.update({"assessment": assessments_to_keep})
        print(f"Updated user assessments. Total assessments: {len(assessments_to_keep)}")
        
        # Return summary statistics
        assessments_removed = len(existing_assessments) - len(assessments_to_keep) + successful_generations
        return {
            "assessments_generated": successful_generations,
            "assessments_kept": len(assessments_to_keep) - successful_generations,
            "assessments_removed": assessments_removed,
            "total_assessments": len(assessments_to_keep)
        }

def process_assessments(user_id: str) -> Dict[str, Any]:
    """Main function to process assessments for a user"""
    try:
        if not db:
            raise Exception("Firebase not initialized")
        
        print(f"Processing assessments for user_id: {user_id}")
        
        # Create assessment service instance
        assessment_service = AssessmentService(db, "pacific-vault-470814-s5")
        
        # Sync assessments with current skills
        stats = assessment_service.sync_assessments_with_skills(user_id)
        
        return {
            "success": True,
            "message": "Assessments synchronized successfully",
            "user_id": user_id,
            **stats
        }
        
    except Exception as e:
        print(f"Error processing assessments for user {user_id}: {str(e)}")
        return {
            "success": False,
            "message": "Failed to process assessments",
            "user_id": user_id,
            "error_details": str(e)
        }

def process_user_id(user_id: str) -> Dict[str, Any]:
    """Main processing function that generates goal and roadmap for a user."""
    try:
        print(f"\nProcessing user_id: {user_id}")
        
        # Retrieve user skills
        print("Step 1: Retrieving user skills...")
        skills_data = get_user_skills(user_id)
        
        if not skills_data:
            raise Exception("No skills data found for user")
        
        print(f"Skills retrieved: {skills_data}")
        
        # Generate career goal
        print("Step 2: Generating career goal...")
        generated_goal = goal_generator(skills_data)
        
        # Generate roadmap
        print("Step 3: Generating roadmap...")
        roadmap = roadmap_generator(skills_data, generated_goal)
        
        # Update Firestore
        print("Step 4: Updating Firestore...")
        update_success = update_user_data(user_id, generated_goal, roadmap)
        
        if update_success:
            return {
                "success": True,
                "message": "Goal and roadmap generated successfully",
                "user_id": user_id,
                "generated_goal": generated_goal,
                "roadmap": roadmap
            }
        else:
            raise Exception("Failed to update Firestore")
            
    except Exception as e:
        print(f"Error processing user {user_id}: {str(e)}")
        return {
            "success": False,
            "message": "Failed to process user",
            "user_id": user_id,
            "error_details": str(e)
        }

@app.post("/process-assessments", response_model=AssessmentResponse)
async def process_user_assessments(request: UserIdRequest):
    """Endpoint to sync assessments with user skills"""
    try:
        if not firebase_initialized:
            raise HTTPException(status_code=500, detail="Firebase initialization failed")
        
        if not request.user_id or not request.user_id.strip():
            raise HTTPException(status_code=400, detail="user_id cannot be empty")
        
        print(f"Received request to process assessments for user_id: {request.user_id}")
        
        # Process assessments
        result = process_assessments(request.user_id.strip())
        
        if result["success"]:
            return AssessmentResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result["error_details"])
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in assessments endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/process-user", response_model=ProcessingResponse)
async def process_user(request: UserIdRequest):
    """Endpoint to process a user_id and generate goal/roadmap"""
    try:
        if not firebase_initialized:
            raise HTTPException(status_code=500, detail="Firebase initialization failed")
        
        if not request.user_id or not request.user_id.strip():
            raise HTTPException(status_code=400, detail="user_id cannot be empty")
        
        print(f"Received request to process user_id: {request.user_id}")
        
        # Process the user
        result = process_user_id(request.user_id.strip())
        
        if result["success"]:
            return ProcessingResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result["error_details"])
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "firebase_initialized": firebase_initialized,
        "message": "Service is running",
        "available_endpoints": [
            "POST /process-user - Generate goal and roadmap",
            "POST /process-assessments - Sync assessments with skills",
            "POST /get-userid - Legacy endpoint",
            "GET /health - Health check",
            "GET /debug-firebase - Debug Firebase connection"
        ]
    }

# Optional: Keep the old endpoint for backward compatibility
@app.post("/get-userid")
async def get_userid(request: UserIdRequest):
    """Legacy endpoint - redirects to the new process-user endpoint"""
    return await process_user(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
