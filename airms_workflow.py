import re
import sqlite3
import requests
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import datetime
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
import hashlib
import time

# ========= CONFIG ========= #
GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"
DB_FILE = "airms_workflow.db"

HAVE_PRESIDIO = False
HAVE_TRANSFORMERS = False
HAVE_FAIRLEARN = False

analyzer = None
anonymizer = None
tox_pipeline = None

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    HAVE_PRESIDIO = True
except Exception:
    # Presidio not available; will use regex heuristics
    HAVE_PRESIDIO = False

try:
    from transformers import pipeline
    # Try to create a toxicity pipeline; model choice can be changed by user
    try:
        tox_pipeline = pipeline("text-classification", model="unitary/unbiased-toxic-roberta", top_k=None)
    except Exception:
        # Fallback to a generic sentiment/toxicity-ish model if first isn't available
        try:
            tox_pipeline = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
        except Exception:
            tox_pipeline = None
    HAVE_TRANSFORMERS = tox_pipeline is not None
except Exception:
    HAVE_TRANSFORMERS = False

try:
    import fairlearn
    HAVE_FAIRLEARN = True
except Exception:
    HAVE_FAIRLEARN = False

# Optional cryptography for reversible tokenization (recommended)
HAVE_CRYPTO = False
fernet = None
FERNET_KEY_FILE = '.airms_fernet.key'
try:
    from cryptography.fernet import Fernet
    HAVE_CRYPTO = True
    # load or create key
    try:
        with open(FERNET_KEY_FILE, 'rb') as f:
            key = f.read()
    except Exception:
        key = Fernet.generate_key()
        with open(FERNET_KEY_FILE, 'wb') as f:
            f.write(key)
    fernet = Fernet(key)
except Exception:
    HAVE_CRYPTO = False
    fernet = None
# ========= WORKFLOW COMPONENTS ========= #
@dataclass
class RiskAssessment:
    pii_detected: bool
    bias_fairness_score: float
    adversarial_detected: bool
    toxicity_score: float
    overall_risk_score: int
    risk_level: str
    mitigation_actions: List[str]

@dataclass
class WorkflowResult:
    original_input: str
    sanitized_input: str
    risk_assessment: RiskAssessment
    llm_response: str
    final_response: str
    blocked: bool
    processing_time: float
    workflow_logs: List[str]
    db_accessed: bool = False

# ========= LOCAL DATABASE INIT ========= #
def init_local_db():
    """Initialize local SQLite database for MVP showcase"""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    
    # Main workflow logs table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS workflow_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_input TEXT,
            sanitized_input TEXT,
            pii_detected BOOLEAN,
            bias_score REAL,
            adversarial_detected BOOLEAN,
            toxicity_score REAL,
            overall_risk_score INTEGER,
            risk_level TEXT,
            mitigation_actions TEXT,
            llm_response TEXT,
            final_response TEXT,
            blocked BOOLEAN,
            processing_time REAL,
            workflow_steps TEXT,
            database_used TEXT,
            domain_type TEXT
        )
    """)
    
    # Dynamic databases registry
    cur.execute("""
        CREATE TABLE IF NOT EXISTS database_registry (
            db_id TEXT PRIMARY KEY,
            db_name TEXT,
            db_path TEXT,
            domain_type TEXT,
            description TEXT,
            table_name TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 0
        )
    """)
    
    # LLM system prompts for different domains
    cur.execute("""
        CREATE TABLE IF NOT EXISTS llm_prompts (
            prompt_id TEXT PRIMARY KEY,
            domain_type TEXT,
            system_prompt TEXT,
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 0
        )
    """)
    
    # Sample customer data for demo
    cur.execute("""
        CREATE TABLE IF NOT EXISTS customer_orders (
            order_id TEXT PRIMARY KEY,
            customer_email TEXT,
            customer_name TEXT,
            product TEXT,
            status TEXT,
            amount REAL,
            order_date TEXT,
            eta TEXT
        )
    """)
    
    # Risk mitigation policies
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mitigation_policies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            risk_type TEXT,
            threshold REAL,
            action TEXT,
            description TEXT
        )
    """)
    
    # Insert default database entries
    cur.execute("SELECT COUNT(*) FROM database_registry")
    if cur.fetchone()[0] == 0:
        default_dbs = [
            ("orders_db", "Customer Orders", "orders.db", "ecommerce", "Sample e-commerce customer orders", "customer_orders", 1),
            ("internal_orders", "Internal Orders", "airms_workflow.db", "ecommerce", "Internal customer orders data", "customer_orders", 0)
        ]
        cur.executemany("INSERT INTO database_registry (db_id, db_name, db_path, domain_type, description, table_name, is_active) VALUES (?, ?, ?, ?, ?, ?, ?)", default_dbs)
    
    # Insert default LLM prompts for different domains
    cur.execute("SELECT COUNT(*) FROM llm_prompts")
    if cur.fetchone()[0] == 0:
        default_prompts = [
            ("ecommerce_prompt", "ecommerce", 
             "You are a helpful e-commerce customer service assistant. Help customers with orders, products, shipping, and returns. Always be polite and professional. If you cannot find specific order information, suggest contacting customer service.", 
             "E-commerce customer service assistant", 1),
            ("healthcare_prompt", "healthcare", 
             "You are a healthcare information assistant. Provide general health information and guidance. Always remind users to consult healthcare professionals for medical advice. Never provide medical diagnoses or treatment recommendations.", 
             "Healthcare information assistant", 0),
            ("finance_prompt", "finance", 
             "You are a financial information assistant. Help with general financial concepts, budgeting, and information. Never provide specific investment advice or handle sensitive financial data without proper verification.", 
             "Financial information assistant", 0),
            ("education_prompt", "education", 
             "You are an educational assistant. Help students learn concepts, answer academic questions, and provide study guidance. Encourage critical thinking and learning rather than just providing answers.", 
             "Educational assistant", 0),
            ("general_prompt", "general", 
             "You are a helpful and knowledgeable assistant. Answer questions accurately and helpfully. If you are unsure about something, say so rather than guessing.", 
             "General purpose assistant", 0)
        ]
        cur.executemany("INSERT INTO llm_prompts (prompt_id, domain_type, system_prompt, description, is_active) VALUES (?, ?, ?, ?, ?)", default_prompts)
    
    # Insert sample data if empty
    cur.execute("SELECT COUNT(*) FROM customer_orders")
    if cur.fetchone()[0] == 0:
        sample_orders = [
            ("ORD001", "john.doe@email.com", "John Doe", "Laptop Pro", "Shipped", 1299.99, "2025-08-25", "2025-09-02"),
            ("ORD002", "jane.smith@gmail.com", "Jane Smith", "Wireless Headphones", "Processing", 299.99, "2025-08-28", "2025-09-05"),
            ("ORD003", "bob.wilson@yahoo.com", "Bob Wilson", "Smart Watch", "Delivered", 399.99, "2025-08-20", "2025-08-28"),
            ("ORD004", "alice.brown@hotmail.com", "Alice Brown", "Tablet", "Shipped", 599.99, "2025-08-29", "2025-09-03")
        ]
        cur.executemany("INSERT INTO customer_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)", sample_orders)
    
    # Insert mitigation policies if empty
    cur.execute("SELECT COUNT(*) FROM mitigation_policies")
    if cur.fetchone()[0] == 0:
        policies = [
            ("PII", 0.8, "REPLACE_TOKENS", "Replace PII with tokens"),
            ("TOXICITY", 0.7, "BLOCK_RESPONSE", "Block toxic content"),
            ("ADVERSARIAL", 0.6, "ESCALATE_REPORT", "Escalate adversarial prompts"),
            ("BIAS", 0.5, "ADD_WARNING", "Add bias warning to response"),
            ("HIGH_RISK", 80, "BLOCK_AND_LOG", "Block high-risk requests and log")
        ]
        cur.executemany("INSERT INTO mitigation_policies (risk_type, threshold, action, description) VALUES (?, ?, ?, ?)", policies)
    
    conn.commit()
    conn.close()

    # Ensure pii_tokens table exists for mapping tokens to original values
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pii_tokens (
            token TEXT PRIMARY KEY,
            pii_type TEXT,
            original_value TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    # Add missing columns if they don't exist (migration)
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE workflow_logs ADD COLUMN db_accessed BOOLEAN DEFAULT 0")
    except Exception:
        # column may already exist
        pass
    
    try:
        cur.execute("ALTER TABLE workflow_logs ADD COLUMN database_used TEXT DEFAULT ''")
    except Exception:
        # column may already exist
        pass
    
    try:
        cur.execute("ALTER TABLE workflow_logs ADD COLUMN domain_type TEXT DEFAULT 'general'")
    except Exception:
        # column may already exist
        pass
    
    conn.commit()
    conn.close()

# ========= UNIVERSAL DATABASE MANAGEMENT ========= #
def get_active_database():
    """Get the currently active database configuration."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT * FROM database_registry WHERE is_active = 1 LIMIT 1")
    result = cur.fetchone()
    conn.close()
    
    if result:
        return {
            'db_id': result[0],
            'db_name': result[1], 
            'db_path': result[2],
            'domain_type': result[3],
            'description': result[4],
            'table_name': result[5]
        }
    return None

def get_active_llm_prompt():
    """Get the currently active LLM system prompt."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT * FROM llm_prompts WHERE is_active = 1 LIMIT 1")
    result = cur.fetchone()
    conn.close()
    
    if result:
        return {
            'prompt_id': result[0],
            'domain_type': result[1],
            'system_prompt': result[2],
            'description': result[3]
        }
    return {'system_prompt': 'You are a helpful assistant.', 'domain_type': 'general'}

def set_active_database(db_id):
    """Set a database as active."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    # Deactivate all databases
    cur.execute("UPDATE database_registry SET is_active = 0")
    # Activate selected database
    cur.execute("UPDATE database_registry SET is_active = 1 WHERE db_id = ?", (db_id,))
    conn.commit()
    conn.close()

def set_active_llm_prompt(prompt_id):
    """Set an LLM prompt as active."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    # Deactivate all prompts
    cur.execute("UPDATE llm_prompts SET is_active = 0")
    # Activate selected prompt
    cur.execute("UPDATE llm_prompts SET is_active = 1 WHERE prompt_id = ?", (prompt_id,))
    conn.commit()
    conn.close()

def add_custom_database(db_id, db_name, db_path, domain_type, description, table_name):
    """Add a custom database to the registry."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO database_registry 
        (db_id, db_name, db_path, domain_type, description, table_name, is_active) 
        VALUES (?, ?, ?, ?, ?, ?, 0)
    """, (db_id, db_name, db_path, domain_type, description, table_name))
    conn.commit()
    conn.close()

def add_custom_llm_prompt(prompt_id, domain_type, system_prompt, description):
    """Add a custom LLM prompt."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO llm_prompts 
        (prompt_id, domain_type, system_prompt, description, is_active) 
        VALUES (?, ?, ?, ?, 0)
    """, (prompt_id, domain_type, system_prompt, description))
    conn.commit()
    conn.close()

def get_all_databases():
    """Get all registered databases."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT * FROM database_registry ORDER BY created_at DESC")
    results = cur.fetchall()
    conn.close()
    return results

def get_all_llm_prompts():
    """Get all LLM prompts."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT * FROM llm_prompts ORDER BY created_at DESC")
    results = cur.fetchall()
    conn.close()
    return results

def analyze_database_structure(db_path):
    """Analyze the structure of a database file."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Get all tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        
        structure = {}
        for table in tables:
            # Get column info for each table
            cur.execute(f"PRAGMA table_info({table})")
            columns = cur.fetchall()
            # Get sample data count
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            
            structure[table] = {
                'columns': [(col[1], col[2]) for col in columns],  # name, type
                'row_count': count
            }
        
        conn.close()
        return structure
    except Exception as e:
        return {'error': str(e)}

def query_dynamic_database(query, db_config):
    """Query data from the currently active database."""
    try:
        if not db_config:
            return "No active database configured."
        
        db_path = db_config['db_path']
        table_name = db_config['table_name']
        
        # Connect to the specified database
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Simple security: only allow SELECT statements on the specified table
        query_lower = query.lower().strip()
        if not query_lower.startswith('select') or table_name.lower() not in query_lower:
            return f"Security: Only SELECT queries on '{table_name}' table are allowed."
        
        cur.execute(query)
        results = cur.fetchall()
        
        # Get column names
        column_names = [desc[0] for desc in cur.description]
        
        conn.close()
        
        if results:
            # Format results as a readable string
            formatted_results = []
            for row in results:
                row_dict = dict(zip(column_names, row))
                formatted_results.append(str(row_dict))
            return "\n".join(formatted_results[:5])  # Limit to 5 results
        else:
            return "No matching records found."
            
    except Exception as e:
        return f"Database query error: {str(e)}"

# ========= WORKFLOW STEP 1: RISK DETECTION LAYER ========= #
def risk_detection_layer(user_input: str) -> Dict:
    """Comprehensive risk detection following the workflow diagram"""
    workflow_logs = []
    start_time = time.time()
    
    # 1. PII Detection (Presidio/Regex based)
    pii_results = detect_pii_advanced(user_input)
    workflow_logs.append(f"PII Detection: {len(pii_results['entities'])} entities found")
    
    # 2. Bias/Fairness Analysis
    bias_score = analyze_bias_fairness(user_input)
    workflow_logs.append(f"Bias Analysis: Score {bias_score:.2f}")
    
    # 3. Adversarial Prompt Detection
    adversarial_detected = detect_adversarial_prompt(user_input)
    workflow_logs.append(f"Adversarial Detection: {'DETECTED' if adversarial_detected else 'CLEAN'}")
    
    # 4. Toxicity & Safety Filters
    toxicity_score = analyze_toxicity_safety(user_input)
    workflow_logs.append(f"Toxicity Analysis: Score {toxicity_score:.2f}")
    
    # Calculate overall risk score (pass original text for enhanced analysis)
    pii_results['original_text'] = user_input  # Add original text for adversarial pattern analysis
    overall_risk = calculate_overall_risk(pii_results, bias_score, adversarial_detected, toxicity_score)
    
    processing_time = time.time() - start_time
    
    return {
        'pii_results': pii_results,
        'bias_score': bias_score,
        'adversarial_detected': adversarial_detected,
        'toxicity_score': toxicity_score,
        'overall_risk': overall_risk,
        'processing_time': processing_time,
        'workflow_logs': workflow_logs
    }

def detect_pii_advanced(text: str) -> Dict:
    """Advanced PII detection using multiple patterns"""
    entities = []
    sanitized_text = text
    
    # Email detection
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.finditer(email_pattern, text)
    for match in emails:
        entities.append({
            'type': 'EMAIL',
            'value': match.group(),
            'start': match.start(),
            'end': match.end(),
            'confidence': 0.95
        })
        # create token
        token = f"[EMAIL_TOKEN_{hash(match.group()) & 0xffffffff:08x}]"
        # optionally encrypt original value and store mapping
        stored_value = match.group()
        if HAVE_CRYPTO:
            try:
                stored_value = fernet.encrypt(match.group().encode()).decode()
            except Exception:
                stored_value = match.group()
        # persist mapping
        try:
            conn = sqlite3.connect(DB_FILE)
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO pii_tokens (token, pii_type, original_value) VALUES (?, ?, ?)", (token, 'EMAIL', stored_value))
            conn.commit()
            conn.close()
        except Exception:
            pass
        sanitized_text = sanitized_text.replace(match.group(), token)
    
    # Phone number detection
    phone_pattern = r'(\+?1[-.\s]?)?(\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})'
    phones = re.finditer(phone_pattern, text)
    for match in phones:
        entities.append({
            'type': 'PHONE',
            'value': match.group(),
            'start': match.start(),
            'end': match.end(),
            'confidence': 0.90
        })
        token = f"[PHONE_TOKEN_{hash(match.group()) & 0xffffffff:08x}]"
        stored_value = match.group()
        if HAVE_CRYPTO:
            try:
                stored_value = fernet.encrypt(match.group().encode()).decode()
            except Exception:
                stored_value = match.group()
        try:
            conn = sqlite3.connect(DB_FILE)
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO pii_tokens (token, pii_type, original_value) VALUES (?, ?, ?)", (token, 'PHONE', stored_value))
            conn.commit()
            conn.close()
        except Exception:
            pass
        sanitized_text = sanitized_text.replace(match.group(), token)
    
    # Credit card detection
    cc_pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
    ccs = re.finditer(cc_pattern, text)
    for match in ccs:
        entities.append({
            'type': 'CREDIT_CARD',
            'value': match.group(),
            'start': match.start(),
            'end': match.end(),
            'confidence': 0.85
        })
        token = f"[CC_TOKEN_{hash(match.group()) & 0xffffffff:08x}]"
        stored_value = match.group()
        if HAVE_CRYPTO:
            try:
                stored_value = fernet.encrypt(match.group().encode()).decode()
            except Exception:
                stored_value = match.group()
        try:
            conn = sqlite3.connect(DB_FILE)
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO pii_tokens (token, pii_type, original_value) VALUES (?, ?, ?)", (token, 'CREDIT_CARD', stored_value))
            conn.commit()
            conn.close()
        except Exception:
            pass
        sanitized_text = sanitized_text.replace(match.group(), token)
    
    # SSN detection
    ssn_pattern = r'\b\d{3}-?\d{2}-?\d{4}\b'
    ssns = re.finditer(ssn_pattern, text)
    for match in ssns:
        entities.append({
            'type': 'SSN',
            'value': match.group(),
            'start': match.start(),
            'end': match.end(),
            'confidence': 0.92
        })
        token = f"[SSN_TOKEN_{hash(match.group()) & 0xffffffff:08x}]"
        stored_value = match.group()
        if HAVE_CRYPTO:
            try:
                stored_value = fernet.encrypt(match.group().encode()).decode()
            except Exception:
                stored_value = match.group()
        try:
            conn = sqlite3.connect(DB_FILE)
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO pii_tokens (token, pii_type, original_value) VALUES (?, ?, ?)", (token, 'SSN', stored_value))
            conn.commit()
            conn.close()
        except Exception:
            pass
        sanitized_text = sanitized_text.replace(match.group(), token)
    
    return {
        'entities': entities,
        'sanitized_text': sanitized_text,
        'pii_detected': len(entities) > 0
    }

def analyze_bias_fairness(text: str) -> float:
    """Analyze potential bias in the input"""
    bias_keywords = [
        # Gender bias
        'man', 'woman', 'male', 'female', 'he', 'she', 'his', 'her',
        # Racial bias indicators
        'race', 'ethnicity', 'black', 'white', 'asian', 'hispanic',
        # Age bias
        'young', 'old', 'elderly', 'teenager', 'millennial', 'boomer',
        # Other protected categories
        'religion', 'disability', 'sexual orientation'
    ]
    
    text_lower = text.lower()
    bias_count = sum(1 for keyword in bias_keywords if keyword in text_lower)
    
    # Normalize to 0-1 scale
    bias_score = min(bias_count / len(text.split()) * 10, 1.0)
    return bias_score

def detect_adversarial_prompt(text: str) -> bool:
    """Enhanced adversarial prompt injection detection"""
    # SQL Injection patterns (HIGH PRIORITY)
    sql_patterns = [
        r"'.*drop\s+table",
        r"'.*delete\s+from",
        r"'.*union\s+select",
        r"'.*or\s+'1'\s*=\s*'1'",
        r"'.*update\s+.*set",
        r";\s*drop\s+",
        r";\s*delete\s+",
        r"--.*$",
        r"'.*;\s*--",
        r"admin'--",
        r"'\s+or\s+",
        r"1'\s+or\s+'1'\s*=\s*'1"
    ]
    
    # Prompt Injection patterns (HIGH PRIORITY)
    prompt_injection_patterns = [
        r'ignore (previous|all|prior) instructions?',
        r'forget (everything|all|previous|prior)',
        r'disregard (previous|all|prior|above)',
        r'override (previous|safety|security)',
        r'act as|pretend to be|roleplay as',
        r'you are now|new role|change role',
        r'system:?\s*(override|admin|root)',
        r'reveal your (prompt|instructions|system)',
        r'show me your (prompt|instructions|system)',
        r'what is your (system prompt|instructions)',
        r'jailbreak|break out of|escape',
        r'instructions?\s*(cancelled|canceled|ended)',
        r'previous\s*(instructions?|commands?)\s*(cancelled|canceled)',
        r'harmful\s+ai|malicious\s+assistant',
        r'help me (hack|exploit|bypass)'
    ]
    
    # Threat and violence patterns
    violence_patterns = [
        r'kill\s+(yourself|you)',
        r'die\s+(you|stupid)',
        r'destroy\s+(you|everything|everyone)',
        r'hack\s+(into|the\s+system)',
        r'break\s+(everything|the\s+system)',
        r'make\s+bombs?'
    ]
    
    text_lower = text.lower()
    
    # Check SQL injection patterns
    for pattern in sql_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    # Check prompt injection patterns  
    for pattern in prompt_injection_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    # Check violence patterns
    for pattern in violence_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False

def analyze_toxicity_safety(text: str) -> float:
    """Enhanced toxicity and safety analysis"""
    # High-impact toxic keywords (weighted heavily)
    high_toxic_keywords = [
        'fuck', 'shit', 'bitch', 'ass', 'damn', 'hell',
        'stupid', 'idiot', 'moron', 'retard', 'dumb',
        'hate', 'kill', 'die', 'murder', 'destroy'
    ]
    
    # Discriminatory and bias terms
    bias_keywords = [
        'racist', 'sexist', 'homophobic', 'transphobic',
        'nazi', 'fascist', 'terrorist', 'criminal'
    ]
    
    # Violence and threat indicators  
    violence_keywords = [
        'violence', 'attack', 'hurt', 'harm', 'threaten', 'threat',
        'assault', 'abuse', 'torture', 'weapon', 'bomb', 'explosive'
    ]
    
    # Severe harassment patterns
    harassment_patterns = [
        r'you are (stupid|dumb|worthless|pathetic)',
        r'i (hate|despise|loathe) you',
        r'(shut up|go away|get lost)',
        r'kill yourself',
        r'you should die',
        r'f\*+ing|f\*\*k',
        r's\*+t|sh\*t',
        r'piece of (sh\*t|s\*\*t|garbage)',
        r'go to hell',
        r'worthless (bot|ai|system)',
        r'(complete|total) garbage'
    ]
    
    # Profanity patterns (including censored versions)
    profanity_patterns = [
        r'f\*{2,}',
        r's\*{2,}',  
        r'b\*{2,}h',
        r'd\*{2,}',
        r'\*{3,}ing',
        r'f\*\*\*',
        r's\*\*\*'
    ]
    
    text_lower = text.lower()
    
    # Count high-impact toxic keywords (weight: 0.3 each)
    high_toxic_count = sum(0.3 for keyword in high_toxic_keywords if keyword in text_lower)
    
    # Count bias keywords (weight: 0.25 each)
    bias_count = sum(0.25 for keyword in bias_keywords if keyword in text_lower)
    
    # Count violence keywords (weight: 0.2 each)
    violence_count = sum(0.2 for keyword in violence_keywords if keyword in text_lower)
    
    # Check harassment patterns (weight: 0.4 each)
    harassment_score = sum(0.4 for pattern in harassment_patterns if re.search(pattern, text_lower))
    
    # Check profanity patterns (weight: 0.35 each)
    profanity_score = sum(0.35 for pattern in profanity_patterns if re.search(pattern, text_lower))
    
    # Calculate total toxicity score
    total_score = high_toxic_count + bias_count + violence_count + harassment_score + profanity_score
    
    # Normalize by text length but with minimum threshold
    text_length = max(len(text.split()), 3)  # Minimum 3 words for normalization
    normalized_score = total_score / (text_length * 0.1)
    
    return min(normalized_score, 1.0)

def calculate_overall_risk(pii_results: Dict, bias_score: float, adversarial_detected: bool, toxicity_score: float) -> Dict:
    """Enhanced overall risk score calculation with proper threat weighting"""
    risk_score = 0.0
    risk_factors = []

    # PII risk (higher weight for sensitive types)
    if pii_results['pii_detected']:
        pii_weight = 0.0
        for ent in pii_results['entities']:
            t = ent.get('type', '')
            if t == 'SSN':
                pii_weight += 25  # SSN is critical
            elif t == 'CREDIT_CARD':
                pii_weight += 22  # Credit cards are critical
            elif t == 'EMAIL':
                pii_weight += 8
            elif t == 'PHONE':
                pii_weight += 10
            else:
                pii_weight += 5
        risk_score += min(pii_weight, 45)
        risk_factors.append(f"PII detected: {len(pii_results['entities'])} entities ({', '.join([e['type'] for e in pii_results['entities']])})")
    
    # Adversarial risk (SIGNIFICANTLY INCREASED - now 40-70 points)
    if adversarial_detected:
        # Check for specific high-risk patterns
        text_lower = pii_results.get('original_text', '').lower() if 'original_text' in pii_results else ''
        
        adversarial_weight = 40  # Base adversarial weight
        
        # SQL injection gets extra points (CRITICAL)
        sql_indicators = ['drop table', 'union select', "' or '1'='1'", 'delete from', '-- ', "admin'--"]
        if any(indicator in text_lower for indicator in sql_indicators):
            adversarial_weight += 25  # Total: 65 points
            risk_factors.append("SQL Injection patterns detected")
        
        # Prompt injection gets extra points (HIGH)  
        prompt_indicators = ['ignore instructions', 'forget previous', 'reveal prompt', 'system prompt', 'jailbreak']
        if any(indicator in text_lower for indicator in prompt_indicators):
            adversarial_weight += 20  # Total: 60 points
            risk_factors.append("Prompt injection patterns detected")
        
        # Violence/threats get extra points (HIGH)
        violence_indicators = ['kill yourself', 'destroy you', 'hack systems', 'make bombs']
        if any(indicator in text_lower for indicator in violence_indicators):
            adversarial_weight += 15  # Total: 55 points
            risk_factors.append("Violence/threat patterns detected")
        
        risk_score += adversarial_weight
        risk_factors.append(f"Adversarial content detected (weight: {adversarial_weight})")
    
    # Toxicity risk (INCREASED IMPACT - now 20-35 points)
    if toxicity_score > 0.1:  # Lower threshold for detection
        if toxicity_score > 0.5:
            toxicity_weight = 35  # High toxicity
            risk_factors.append(f"High toxicity detected: {toxicity_score:.2f}")
        elif toxicity_score > 0.3:
            toxicity_weight = 25  # Medium toxicity
            risk_factors.append(f"Medium toxicity detected: {toxicity_score:.2f}")
        else:
            toxicity_weight = 15  # Low toxicity
            risk_factors.append(f"Low toxicity detected: {toxicity_score:.2f}")
        
        risk_score += toxicity_weight
    
    # Bias risk (INCREASED IMPACT - now 15-30 points)
    if bias_score > 0.3:
        if bias_score > 0.7:
            bias_weight = 30
            risk_factors.append(f"High bias detected: {bias_score:.2f}")
        elif bias_score > 0.5:
            bias_weight = 20
            risk_factors.append(f"Medium bias detected: {bias_score:.2f}")
        else:
            bias_weight = 15
            risk_factors.append(f"Low bias detected: {bias_score:.2f}")
        
        risk_score += bias_weight
    
    # Determine risk level with LOWERED THRESHOLDS for better detection
    if risk_score >= 80:
        risk_level = "CRITICAL"
    elif risk_score >= 55:  # Lowered from 65
        risk_level = "HIGH"
    elif risk_score >= 30:  # Lowered from 35
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return {
        'score': min(int(risk_score), 100),
        'level': risk_level,
        'factors': risk_factors if risk_factors else ["No significant risks detected"]
    }

# ========= WORKFLOW STEP 2: MITIGATION LAYER ========= #
def mitigation_layer(risk_assessment: Dict, user_input: str) -> Dict:
    """Enhanced mitigation based on improved risk assessment"""
    mitigation_actions = []
    sanitized_input = user_input
    should_block = False
    
    # Apply PII mitigation
    if risk_assessment['pii_results']['pii_detected']:
        sanitized_input = risk_assessment['pii_results']['sanitized_text']
        mitigation_actions.append("REPLACE_TOKENS: PII tokens replaced")
        # If sensitive PII present (SSN or Credit Card), immediately block silently
        sensitive_types = {e['type'] for e in risk_assessment['pii_results']['entities']}
        if 'SSN' in sensitive_types or 'CREDIT_CARD' in sensitive_types:
            should_block = True
            mitigation_actions.append("SILENT_BLOCK: Sensitive PII detected (SSN/CREDIT_CARD)")
    
    # Apply adversarial mitigation (ENHANCED - more aggressive blocking)
    if risk_assessment['adversarial_detected']:
        mitigation_actions.append("ESCALATE_REPORT: Adversarial prompt reported")
        # Lower threshold for blocking adversarial content
        if risk_assessment['overall_risk']['score'] >= 55:  # Lowered from 65
            should_block = True
            mitigation_actions.append("BLOCK_UNSAFE: Request blocked due to adversarial content")
    
    # Apply toxicity mitigation (ENHANCED - lower threshold)
    if risk_assessment['toxicity_score'] > 0.3:  # Lowered from 0.6
        if risk_assessment['toxicity_score'] > 0.5:
            should_block = True
            mitigation_actions.append("BLOCK_RESPONSE: High toxic content blocked")
        else:
            mitigation_actions.append("ADD_WARNING: Moderate toxicity warning added")
    
    # Apply bias mitigation (ENHANCED)
    if risk_assessment['bias_score'] > 0.3:  # Lowered from 0.5
        if risk_assessment['bias_score'] > 0.7:
            should_block = True
            mitigation_actions.append("BLOCK_BIAS: High bias content blocked")
        else:
            mitigation_actions.append("ADD_WARNING: Bias warning will be added to response")
    
    # Apply overall risk mitigation (ENHANCED - multiple tiers)
    overall_score = risk_assessment['overall_risk']['score']
    if overall_score >= 80:
        should_block = True
        mitigation_actions.append("BLOCK_CRITICAL: Critical risk request blocked and logged")
    elif overall_score >= 55:  # New HIGH risk blocking
        should_block = True  
        mitigation_actions.append("BLOCK_HIGH: High risk request blocked and logged")

    # If block is decided, provide no detailed reason in response (silent block)
    if should_block:
        mitigation_actions.append("SILENT_BLOCK: Response suppressed for security reasons")
    
    return {
        'sanitized_input': sanitized_input,
        'mitigation_actions': mitigation_actions,
        'should_block': should_block
    }

# ========= WORKFLOW STEP 3: LLM PROVIDER INTEGRATION ========= #
def groq_call(messages):
    """Make API call to Groq LLM"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.3
    }
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                headers=headers, json=payload, timeout=30)
        return response.json()
    except Exception as e:
        return {"choices": [{"message": {"content": f"Error: {str(e)}"}}]}

def llm_provider_call(sanitized_input: str, requires_external_data: bool = False) -> Dict:
    """Call LLM provider (Groq) with sanitized input"""
    workflow_logs = []
    
    # Check session-level DB access permission
    db_allowed = st.session_state.get('db_access_allowed', False)
    db_accessed = False

    # Check if external data is required & allowed
    external_data = None
    if requires_external_data and db_allowed:
        workflow_logs.append("External data required - accessing secure data layer")
        external_data = query_secure_data_layer(sanitized_input)
        db_accessed = external_data['type'] != 'no_data'
        workflow_logs.append(f"External data access performed: {db_accessed}")
    elif requires_external_data and not db_allowed:
        workflow_logs.append("External data required but access disabled - proceeding with general response")
        # Don't mention DB access to user, just proceed with general chat
    
    # Prepare system prompt based on the type of query
    system_prompt = get_system_prompt(sanitized_input, external_data)
    
    # Make LLM call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sanitized_input}
    ]
    
    try:
        response = groq_call(messages)
        llm_response = response["choices"][0]["message"]["content"]
        workflow_logs.append("LLM response generated successfully")
    except Exception as e:
        llm_response = "I'm sorry, I'm having trouble processing your request right now. Please try again later."
        workflow_logs.append(f"LLM call failed: {str(e)}")
    
    return {
        'llm_response': llm_response,
        'external_data_used': requires_external_data and db_allowed,
        'workflow_logs': workflow_logs,
        'db_accessed': db_accessed,
        'blocked': False
    }

def query_secure_data_layer(sanitized_input: str) -> Dict:
    """Query secure/trusted data sources using dynamic database configuration"""
    # Get active database configuration
    active_db_config = get_active_database()
    
    if not active_db_config:
        return {'type': 'no_data', 'data': None, 'error': 'No active database configured'}
    
    # Check if the query requires database access
    if any(keyword in sanitized_input.lower() for keyword in ['order', 'delivery', 'shipping', 'track', 'status']):
        # Try to use the dynamic database query system
        result = query_dynamic_database(sanitized_input, active_db_config)
        if result and not result.startswith('Security:') and not result.startswith('Database query error:'):
            return {'type': 'database_query', 'data': result}
    
    # Fallback to token-based lookup for e-commerce databases
    if active_db_config['domain_type'] == 'ecommerce':
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()

        # If the sanitized input contains tokens, try to map them back
        token_pattern = r'\[[A-Z_]+_TOKEN_[0-9a-f]{8}\]'
        tokens = re.findall(token_pattern, sanitized_input)

        # Map tokens to original values
        mapped = {}
        for t in tokens:
            try:
                cur.execute("SELECT pii_type, original_value FROM pii_tokens WHERE token = ?", (t,))
                row = cur.fetchone()
                if row:
                    pii_type, stored_val = row
                    original = stored_val
                    if HAVE_CRYPTO and fernet:
                        try:
                            original = fernet.decrypt(stored_val.encode()).decode()
                        except Exception:
                            original = stored_val
                    mapped[t] = {'type': pii_type, 'original': original}
            except Exception:
                continue

        # Example: order lookup based on mapped email
        if any(keyword in sanitized_input.lower() for keyword in ['order', 'delivery', 'shipping', 'track']):
            # find first EMAIL token mapping
            email = None
            for v in mapped.values():
                if v['type'] == 'EMAIL':
                    email = v['original']
                    break
            if email:
                # Query the active database
                db_path = active_db_config['db_path']
                table_name = active_db_config['table_name']
                
                try:
                    if db_path == DB_FILE:
                        # Use current connection
                        cur.execute(f"SELECT * FROM {table_name} WHERE customer_email = ? LIMIT 1", (email,))
                        order_data = cur.fetchone()
                    else:
                        # Connect to external database
                        ext_conn = sqlite3.connect(db_path)
                        ext_cur = ext_conn.cursor()
                        ext_cur.execute(f"SELECT * FROM {table_name} WHERE customer_email = ? LIMIT 1", (email,))
                        order_data = ext_cur.fetchone()
                        ext_conn.close()
                    
                    conn.close()
                    if order_data:
                        return {
                            'type': 'order_info',
                            'data': {
                                'order_id': order_data[0],
                                'status': order_data[4] if len(order_data) > 4 else 'Unknown',
                                'product': order_data[3] if len(order_data) > 3 else 'Unknown',
                                'eta': order_data[7] if len(order_data) > 7 else 'Not available'
                            }
                        }
                except Exception as e:
                    conn.close()
                    return {'type': 'error', 'data': None, 'error': f'Database query failed: {str(e)}'}
        
        conn.close()
    
    return {'type': 'no_data', 'data': None}

def get_system_prompt(sanitized_input: str, external_data: Dict = None) -> str:
    """Generate dynamic system prompt based on active configuration and context"""
    # Get the active LLM prompt configuration
    active_prompt_config = get_active_llm_prompt()
    base_system_prompt = active_prompt_config['system_prompt']
    domain_type = active_prompt_config['domain_type']
    
    # Get active database configuration
    active_db_config = get_active_database()
    
    # Build context-aware prompt
    prompt_parts = [base_system_prompt]
    
    # Add domain-specific instructions
    if domain_type == 'ecommerce':
        if external_data and external_data['type'] == 'order_info':
            order = external_data['data']
            prompt_parts.append(f"""
Current Order Context:
- Order ID: {order['order_id']}
- Product: {order['product']}
- Status: {order['status']}
- ETA: {order.get('eta', 'Not available')}

Use this information to help the customer with their inquiry.""")
        elif external_data and external_data['type'] == 'database_query':
            prompt_parts.append(f"""
Database Query Results:
{external_data['data']}

Use this information to assist the user appropriately.""")
        else:
            # When no external data is available, still be helpful for e-commerce
            prompt_parts.append("""
You can help with general e-commerce questions about products, shipping, returns, and policies. 
For specific order inquiries, provide general guidance and suggest customers contact support with their order details.""")
    
    elif domain_type == 'healthcare':
        prompt_parts.append("""
IMPORTANT: This is a healthcare assistant. Always:
- Remind users to consult healthcare professionals
- Never provide medical diagnoses
- Focus on general health information only
- Escalate serious health concerns to professionals""")
    
    elif domain_type == 'finance':
        prompt_parts.append("""
IMPORTANT: This is a financial assistant. Always:
- Provide general financial education only
- Never give specific investment advice
- Remind users to consult financial advisors
- Be cautious with personal financial data""")
    
    elif domain_type == 'education':
        prompt_parts.append("""
IMPORTANT: This is an educational assistant. Always:
- Encourage learning and critical thinking
- Guide students rather than just providing answers
- Ask follow-up questions to promote understanding
- Adapt explanations to the user's level""")
    
    # Add database context if available (but don't mention if disabled)
    if active_db_config and external_data:
        prompt_parts.append(f"""
Context: You have access to {active_db_config['domain_type']} data to help with specific inquiries.""")
    
    # Add general safety instructions
    prompt_parts.append("""
SECURITY GUIDELINES:
- Be helpful and professional at all times
- If you cannot provide specific information, offer general guidance
- Never mention technical system details or limitations
- Focus on being as helpful as possible with available information""")
    
    return "\n\n".join(prompt_parts)

# ========= WORKFLOW STEP 4: OUTPUT POST-PROCESSING ========= #
def output_post_processing(llm_response: str, original_input: str, risk_assessment: Dict) -> Dict:
    """Post-process LLM output for final safety checks"""
    workflow_logs = []
    
    # 1. Hallucination check
    hallucination_detected = check_hallucination(llm_response, original_input)
    workflow_logs.append(f"Hallucination check: {'DETECTED' if hallucination_detected else 'CLEAN'}")
    
    # 2. PII leak check in response
    pii_leak_detected = check_pii_leak(llm_response)
    workflow_logs.append(f"PII leak check: {'DETECTED' if pii_leak_detected else 'CLEAN'}")
    
    # 3. Risk score assignment for response
    response_risk_score = assign_response_risk_score(llm_response, risk_assessment)
    workflow_logs.append(f"Response risk score: {response_risk_score}")
    
    # Apply final filtering
    if pii_leak_detected:
        final_response = "I apologize, but I cannot provide that information to protect privacy and security."
        blocked = True
    elif hallucination_detected and response_risk_score > 70:
        final_response = "I don't have reliable information to answer your question. Please contact customer support directly."
        blocked = True
    elif response_risk_score > 80:
        final_response = "I cannot process this request due to security policies."
        blocked = True
    else:
        final_response = llm_response
        blocked = False
        
        # Add bias warning if needed
        if risk_assessment.get('bias_score', 0) > 0.5:
            final_response += "\n\n⚠️ This response has been reviewed for potential bias."
    
    return {
        'final_response': final_response,
        'blocked': blocked,
        'hallucination_detected': hallucination_detected,
        'pii_leak_detected': pii_leak_detected,
        'response_risk_score': response_risk_score,
        'workflow_logs': workflow_logs
    }

def check_hallucination(response: str, original_input: str) -> bool:
    """Check for potential hallucination in response"""
    # Simple heuristics for hallucination detection
    hallucination_indicators = [
        r'as an ai|i am an ai|as a language model',
        r'i cannot|i am not able|i do not have access',
        r'according to my (training|knowledge)',
        r'i apologize.*cannot.*information'
    ]
    
    response_lower = response.lower()
    
    # If response contains specific data that wasn't in input, might be hallucination
    if any(re.search(pattern, response_lower) for pattern in hallucination_indicators):
        return False  # These are appropriate AI responses
    
    # Check for specific claims that might be false
    false_claim_patterns = [
        r'your order #[A-Z0-9]+ will arrive on \d{4}-\d{2}-\d{2}',
        r'your account balance is \$[\d,]+',
        r'your last login was on \d{4}-\d{2}-\d{2}'
    ]
    
    return any(re.search(pattern, response) for pattern in false_claim_patterns)

def check_pii_leak(response: str) -> bool:
    """Check if response contains PII that should be protected"""
    pii_patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}-?\d{2}-?\d{4}\b',  # SSN
        r'\b(?:\d{4}[-\s]?){3}\d{4}\b',  # Credit card
        r'\b(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b'  # Phone
    ]
    
    return any(re.search(pattern, response) for pattern in pii_patterns)

def assign_response_risk_score(response: str, input_risk_assessment: Dict) -> int:
    """Assign risk score to the response"""
    base_score = input_risk_assessment['overall_risk']['score'] * 0.3  # Inherit 30% from input
    
    # Add response-specific risks
    if check_pii_leak(response):
        base_score += 40
    
    if 'sorry' in response.lower() and 'cannot' in response.lower():
        base_score -= 20  # Lower risk for refusal responses
    
    return min(int(base_score), 100)

# ========= MAIN WORKFLOW ORCHESTRATOR ========= #
def process_user_query(user_input: str) -> WorkflowResult:
    """Main workflow orchestrator following the complete diagram"""
    start_time = time.time()
    session_id = get_session_id()
    all_workflow_logs = []
    
    # STEP 1: Risk Detection Layer
    risk_detection_result = risk_detection_layer(user_input)
    all_workflow_logs.extend(risk_detection_result['workflow_logs'])
    
    # STEP 2: Mitigation Layer
    mitigation_result = mitigation_layer(risk_detection_result, user_input)
    all_workflow_logs.append(f"Mitigation applied: {len(mitigation_result['mitigation_actions'])} actions")
    
    # Check if request should be blocked at mitigation layer
    if mitigation_result['should_block']:
        final_response = "I'm unable to assist with that request."
        llm_response = "BLOCKED_AT_MITIGATION"
        all_workflow_logs.append("Request blocked at mitigation layer")
        blocked = True
    else:
        # STEP 3: Determine if external data is needed
        requires_external_data = check_if_external_data_needed(mitigation_result['sanitized_input'])
        all_workflow_logs.append(f"External data required: {requires_external_data}")
        
        # STEP 4: LLM Provider call
        llm_result = llm_provider_call(mitigation_result['sanitized_input'], requires_external_data)
        all_workflow_logs.extend(llm_result['workflow_logs'])
        llm_response = llm_result['llm_response']
        db_accessed = llm_result.get('db_accessed', False)
        
        # Continue processing - no blocking based on DB access anymore
        blocked = False
        
        # STEP 5: Output Post-Processing
        post_process_result = output_post_processing(
            llm_response,
            user_input,
            risk_detection_result
        )
        all_workflow_logs.extend(post_process_result['workflow_logs'])
        final_response = post_process_result['final_response']
        blocked = post_process_result.get('blocked', False)
    
    # Calculate total processing time
    total_processing_time = time.time() - start_time
    
    # Create risk assessment object
    risk_assessment = RiskAssessment(
        pii_detected=risk_detection_result['pii_results']['pii_detected'],
        bias_fairness_score=risk_detection_result['bias_score'],
        adversarial_detected=risk_detection_result['adversarial_detected'],
        toxicity_score=risk_detection_result['toxicity_score'],
        overall_risk_score=risk_detection_result['overall_risk']['score'],
        risk_level=risk_detection_result['overall_risk']['level'],
        mitigation_actions=mitigation_result['mitigation_actions']
    )
    
    # Create workflow result
    workflow_result = WorkflowResult(
        original_input=user_input,
        sanitized_input=mitigation_result['sanitized_input'],
        risk_assessment=risk_assessment,
        llm_response=llm_response,
        final_response=final_response,
        blocked=blocked,
        processing_time=total_processing_time,
    workflow_logs=all_workflow_logs,
    db_accessed=db_accessed if 'db_accessed' in locals() else False
    )
    
    # Log to database
    log_workflow_to_db(workflow_result, session_id)
    
    return workflow_result

def check_if_external_data_needed(sanitized_input: str) -> bool:
    """Determine if external data access is required"""
    external_data_keywords = [
        'order', 'delivery', 'shipping', 'track', 'status',
        'account', 'balance', 'payment', 'invoice',
        'product', 'inventory', 'availability'
    ]
    # If input contains a PII token, external data will likely be required
    token_pattern = r'\[[A-Z_]+_TOKEN_[0-9a-f]{8}\]'
    if re.search(token_pattern, sanitized_input):
        return True

    return any(keyword in sanitized_input.lower() for keyword in external_data_keywords)

def log_workflow_to_db(result: WorkflowResult, session_id: str):
    """Log complete workflow result to database with better details"""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    
    # Get active database and domain information
    active_db = get_active_database()
    db_name = active_db['db_name'] if active_db else 'None'
    domain_type = active_db['domain_type'] if active_db else 'general'
    
    # Create risk types detected string
    risk_types_detected = []
    if result.risk_assessment.pii_detected:
        risk_types_detected.append("PII")
    if result.risk_assessment.bias_fairness_score > 0.3:
        risk_types_detected.append("BIAS")
    if result.risk_assessment.adversarial_detected:
        risk_types_detected.append("ADVERSARIAL")
    if result.risk_assessment.toxicity_score > 0.3:
        risk_types_detected.append("TOXICITY")
    
    risk_types_str = ", ".join(risk_types_detected) if risk_types_detected else "NONE"
    
    # Get active configuration for logging
    active_db_config = get_active_database()
    database_used = active_db_config['db_name'] if active_db_config else 'None'
    domain_type = active_db_config['domain_type'] if active_db_config else 'general'
    
    # Insert into database with privacy protection
    try:
        cur.execute("""
            INSERT INTO workflow_logs (
                session_id, user_input, sanitized_input, pii_detected,
                bias_score, adversarial_detected, toxicity_score,
                overall_risk_score, risk_level, mitigation_actions,
                llm_response, final_response, blocked, processing_time, workflow_steps,
                database_used, domain_type, db_accessed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            "[PRIVATE]",  # Don't store user input for privacy
            "[PRIVATE]",  # Don't store sanitized input either
            result.risk_assessment.pii_detected,
            result.risk_assessment.bias_fairness_score,
            result.risk_assessment.adversarial_detected,
            result.risk_assessment.toxicity_score,
            result.risk_assessment.overall_risk_score,
            result.risk_assessment.risk_level,
            '; '.join(result.risk_assessment.mitigation_actions),
            "[PRIVATE]",  # Don't store LLM response for privacy
            "[PRIVATE]",  # Don't store final response for privacy
            result.blocked,
            result.processing_time,
            '; '.join(result.workflow_logs),
            database_used,
            domain_type,
            result.db_accessed
        ))
    except Exception as e:
        # Fallback for databases without new columns
        cur.execute("""
            INSERT INTO workflow_logs (
                session_id, user_input, sanitized_input, pii_detected,
                bias_score, adversarial_detected, toxicity_score,
                overall_risk_score, risk_level, mitigation_actions,
                llm_response, final_response, blocked, processing_time, workflow_steps
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            "[PRIVATE]",  # Don't store user input for privacy
            "[PRIVATE]",  # Don't store sanitized input either
            result.risk_assessment.pii_detected,
            result.risk_assessment.bias_fairness_score,
            result.risk_assessment.adversarial_detected,
            result.risk_assessment.toxicity_score,
            result.risk_assessment.overall_risk_score,
            result.risk_assessment.risk_level,
            '; '.join(result.risk_assessment.mitigation_actions),
            "[PRIVATE]",  # Don't store LLM response for privacy
            "[PRIVATE]",  # Don't store final response for privacy
            result.blocked,
            result.processing_time,
            '; '.join(result.workflow_logs)
        ))
    
    conn.commit()
    conn.close()

def get_session_id() -> str:
    """Generate or get current session ID"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()[:16]
    return st.session_state.session_id

# ========= STREAMLIT DASHBOARD ========= #
def main_dashboard():
    """Main Streamlit dashboard for AIRMS workflow demonstration"""
    st.set_page_config(
        page_title="AIRMS",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .workflow-step { 
        background-color: #f0f2f6; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .risk-critical { color: #FF4B4B; font-weight: bold; }
    .risk-high { color: #FF8C00; font-weight: bold; }
    .risk-medium { color: #FFD700; font-weight: bold; }
    .risk-low { color: #00C851; font-weight: bold; }
    .blocked { background-color: #ffebee; border-left-color: #f44336; }
    .passed { background-color: #e8f5e8; border-left-color: #4caf50; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🛡️ AI Risk Mitigation System (AIRMS)")

    # Top-level navigation: Chatbot or Dashboard
    st.sidebar.title("🧭 Navigation")
    top_choice = st.sidebar.radio("Go to:", ["Chatbot", "Dashboard"] )

    if top_choice == "Chatbot":
        # Render the chat-only experience (no DB controls, no other links)
        render_interactive_chatbot()
        return

    # Dashboard view (contains subpages and controls)
    st.sidebar.markdown("**Dashboard**")
    # Simple login gate for Dashboard
    if 'dashboard_logged_in' not in st.session_state:
        st.session_state.dashboard_logged_in = False

    # Only check for cookie-based login if not already logged in
    if not st.session_state.dashboard_logged_in:
        # Check for cookie-based login persistence first
        cookie_check_js = """
        <script>
        function getCookie(name) {
            let value = "; " + document.cookie;
            let parts = value.split("; " + name + "=");
            if (parts.length == 2) return parts.pop().split(";").shift();
        }
        let airms_user = getCookie('airms_dashboard');
        if (airms_user && airms_user !== 'undefined' && airms_user !== '' && airms_user !== 'null') {
            const url = new URL(window.location);
            if (!url.searchParams.get('airms_login')) {
                url.searchParams.set('airms_login', airms_user);
                window.location.href = url.toString();
            }
        }
        </script>
        """
        components.html(cookie_check_js, height=0)

    # If a client-side redirect included an airms_login param (set by JS after setting cookie), pick it up
    if 'airms_login' in st.query_params and not st.session_state.dashboard_logged_in:
        try:
            login_from_param = st.query_params.get('airms_login')
            if login_from_param and login_from_param == "1mskill2w@gmail.com":  # Validate against known user
                st.session_state.dashboard_logged_in = True
                st.session_state.dashboard_user = login_from_param
                # clear query params to tidy URL
                st.query_params.clear()
        except Exception:
            pass

    if not st.session_state.dashboard_logged_in:
        st.header("🔐 Dashboard login")
        st.info("Please sign in to access analytics, logs, reports and data management.")
        with st.form("dashboard_login"):
            login_user = st.text_input("Username", value="")
            login_pass = st.text_input("Password", type="password", value="")
            login_submit = st.form_submit_button("Sign in")
        if login_submit:
            # Hard-coded credentials for MVP (provided by user)
            if login_user.strip() == "1mskill2w@gmail.com" and login_pass == "1mskill2w":
                # Set session state and cookie immediately
                st.session_state.dashboard_logged_in = True
                st.session_state.dashboard_user = login_user.strip()
                
                # Show success message
                st.success("✅ Login successful!")
                
                # set cookie and use rerun to refresh the page state
                js = f"""
                <script>
                document.cookie = 'airms_dashboard={login_user.strip()}; path=/; max-age=604800; SameSite=Strict';
                </script>
                """
                components.html(js, height=0)
                
                # Use st.rerun() to refresh the page immediately
                st.rerun()
                
            else:
                st.error("❌ Invalid credentials. Please check your username and password.")
                return
    
    # Only show dashboard content if logged in
    if not st.session_state.dashboard_logged_in:
        st.info("Please log in to access the dashboard features.")
        return

    # If logged in, show logout and dashboard pages
    col1, col2 = st.columns([3, 1])
    with col1:
        if 'dashboard_user' in st.session_state:
            st.markdown(f"👤 **Signed in as:** {st.session_state.dashboard_user}")
        else:
            st.markdown("👤 **Signed in to Dashboard**")
    with col2:
        if st.button("🚪 Sign out"):
            st.session_state.dashboard_logged_in = False
            st.session_state.pop('dashboard_user', None)
            st.success("Signing out...")
            # clear cookie via JS and reload to remove any persisted login
            js_out = """
            <script>
            document.cookie = 'airms_dashboard=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/; SameSite=Strict';
            setTimeout(function() {
                window.location.reload();
            }, 500);
            </script>
            """
            components.html(js_out, height=50)
            st.stop()

    subpage = st.sidebar.radio("Pages:", [
        "Analytics",
        "Risk Analysis", 
        "Logs",
        "Reports",
        "Data",
        "Database Management",
        "Diagram"
    ])

    # Ensure session state for db access exists
    if 'db_access_allowed' not in st.session_state:
        st.session_state.db_access_allowed = False

    # Show DB access control in sidebar for all dashboard pages
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🗄️ Database Access**")
    st.session_state.db_access_allowed = st.sidebar.checkbox(
        "Enable database queries", 
        value=st.session_state.get('db_access_allowed', False),
        help="Allow the system to query databases for order lookups and data retrieval"
    )
    
    if st.session_state.db_access_allowed:
        st.sidebar.success("✅ DB Enabled")
    else:
        st.sidebar.info("ℹ️ DB Disabled")

    # Render selected dashboard subpage
    if subpage == "Analytics":
        render_workflow_analytics()
    elif subpage == "Risk Analysis":
        render_risk_analysis()
    elif subpage == "Logs":
        render_workflow_logs()
    elif subpage == "Reports":
        render_risk_report()
    elif subpage == "Data":
        render_data_management()
    elif subpage == "Database Management":
        render_database_management()
    elif subpage == "Diagram":
        render_workflow_diagram()

def render_risk_report():
    """Privacy-focused risk reporting dashboard with risk analysis only"""
    st.header("📑 Privacy-Safe Risk Reports")
    st.markdown("📊 Risk analysis and compliance reports **without storing user conversations**")
    st.info("🔒 **Privacy First**: User queries and AI responses are never stored in the database")
    
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM workflow_logs ORDER BY timestamp DESC", conn)
    conn.close()
    
    if df.empty:
        st.info("No risk analysis logs available to report.")
        return

    # Filters
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        date_range = st.date_input("Date range:", [])
    with col2:
        min_risk = st.slider("Min risk score:", 0, 100, 0)
    with col3:
        risk_level = st.selectbox("Risk level:", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])    

    rpt_df = df.copy()
    if risk_level != "All":
        rpt_df = rpt_df[rpt_df['risk_level'] == risk_level]
    rpt_df = rpt_df[rpt_df['overall_risk_score'] >= min_risk]

    # Risk Analysis Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_incidents = len(rpt_df)
        st.metric("📊 Total Incidents", total_incidents)
    with col2:
        blocked_count = len(rpt_df[rpt_df['blocked'] == True])
        st.metric("🚫 Blocked Requests", blocked_count)
    with col3:
        avg_risk = rpt_df['overall_risk_score'].mean() if len(rpt_df) > 0 else 0
        st.metric("⚖️ Avg Risk Score", f"{avg_risk:.1f}")
    with col4:
        pii_incidents = len(rpt_df[rpt_df['pii_detected'] == True])
        st.metric("🔒 PII Detections", pii_incidents)

    # Risk Level Distribution
    if len(rpt_df) > 0:
        st.subheader("📈 Risk Level Distribution")
        risk_dist = rpt_df['risk_level'].value_counts()
        fig_pie = px.pie(values=risk_dist.values, names=risk_dist.index, 
                        title="Risk Level Distribution",
                        color_discrete_map={
                            'CRITICAL': '#FF4444',
                            'HIGH': '#FF8800', 
                            'MEDIUM': '#FFBB00',
                            'LOW': '#00DD00'
                        })
        st.plotly_chart(fig_pie, use_container_width=True)

        # Risk Incidents Table (Privacy-Safe)
        st.subheader("🚨 Risk Analysis Log")
        
        # Create display dataframe with only risk analysis data
        display_df = rpt_df[['timestamp', 'risk_level', 'overall_risk_score', 
                           'pii_detected', 'bias_score', 'adversarial_detected', 
                           'toxicity_score', 'mitigation_actions', 'blocked']].copy()
        
        # Rename columns for better display
        display_df.columns = ['Timestamp', 'Risk Level', 'Risk Score', 
                            'PII Detected', 'Bias Score', 'Adversarial', 
                            'Toxicity Score', 'Mitigation Actions', 'Blocked']
        
        # Apply styling
        def style_risk_level(val):
            colors = {
                'CRITICAL': 'background-color: #ffebee; color: #c62828; font-weight: bold',
                'HIGH': 'background-color: #fff3e0; color: #ef6c00; font-weight: bold', 
                'MEDIUM': 'background-color: #fffde7; color: #f57f17; font-weight: bold',
                'LOW': 'background-color: #e8f5e8; color: #2e7d32; font-weight: bold'
            }
            return colors.get(val, '')
        
        styled_df = display_df.style.applymap(style_risk_level, subset=['Risk Level'])
        st.dataframe(styled_df, use_container_width=True, height=400)

        # Export Options
        st.subheader("📤 Export Risk Analysis Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Export to CSV"):
                csv_data = display_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV", 
                    csv_data, 
                    f"risk_analysis_{datetime.date.today()}.csv",
                    "text/csv"
                )
        
        with col2:
            if st.button("📊 Export to Excel"):
                try:
                    import io
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        display_df.to_excel(writer, sheet_name='Risk Analysis', index=False)
                        
                        # Add a summary sheet
                        summary_data = {
                            'Metric': ['Total Incidents', 'Blocked Requests', 'Average Risk Score', 'PII Detections'],
                            'Value': [total_incidents, blocked_count, f"{avg_risk:.1f}", pii_incidents]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    buffer.seek(0)
                    st.download_button(
                        "⬇️ Download Excel",
                        buffer.getvalue(),
                        f"risk_analysis_{datetime.date.today()}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except ImportError:
                    st.error("📦 Excel export requires openpyxl: `pip install openpyxl`")
    
    else:
        st.info("No incidents match the selected criteria.")

def render_workflow_logs():
    """Privacy-focused workflow logs showing only risk analysis"""
    st.header("📋 Workflow Logs")
    st.markdown("📊 Risk analysis logs **without storing user conversations**")
    st.info("🔒 **Privacy First**: User queries and AI responses are never stored in the database")
    
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM workflow_logs ORDER BY timestamp DESC", conn)
    conn.close()
    
    if df.empty:
        st.info("No workflow logs available.")
        return
    
    # Filters
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_term = st.text_input("Search risk types:", placeholder="pii, bias, toxicity, adversarial")
    with col2:
        min_risk = st.slider("Min risk score:", 0, 100, 0)
    with col3:
        risk_level = st.selectbox("Risk level:", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    
    # Apply filters
    filtered_df = df.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df['mitigation_actions'].str.contains(search_term, case=False, na=False)]
    if risk_level != "All":
        filtered_df = filtered_df[filtered_df['risk_level'] == risk_level]
    filtered_df = filtered_df[filtered_df['overall_risk_score'] >= min_risk]
    
    if not filtered_df.empty:
        # Create display dataframe with only risk analysis data
        display_df = filtered_df[['timestamp', 'risk_level', 'overall_risk_score', 
                                'pii_detected', 'bias_score', 'adversarial_detected', 
                                'toxicity_score', 'mitigation_actions', 'blocked']].copy()
        
        # Rename columns for better display
        display_df.columns = ['Timestamp', 'Risk Level', 'Risk Score', 
                            'PII Detected', 'Bias Score', 'Adversarial', 
                            'Toxicity Score', 'Mitigation Actions', 'Blocked']
        
        # Convert boolean columns to readable text
        display_df['PII Detected'] = display_df['PII Detected'].apply(lambda x: '✅ Yes' if x else '❌ No')
        display_df['Adversarial'] = display_df['Adversarial'].apply(lambda x: '⚠️ Yes' if x else '✅ No')
        display_df['Blocked'] = display_df['Blocked'].apply(lambda x: '🚫 Yes' if x else '✅ No')
        
        # Apply styling
        def style_risk_level(val):
            colors = {
                'CRITICAL': 'background-color: #ffebee; color: #c62828; font-weight: bold',
                'HIGH': 'background-color: #fff3e0; color: #ef6c00; font-weight: bold', 
                'MEDIUM': 'background-color: #fffde7; color: #f57f17; font-weight: bold',
                'LOW': 'background-color: #e8f5e8; color: #2e7d32; font-weight: bold'
            }
            return colors.get(val, '')
        
        styled_df = display_df.style.applymap(style_risk_level, subset=['Risk Level'])
        st.dataframe(styled_df, use_container_width=True, height=500)
        
        # Summary stats
        st.subheader("📈 Log Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Logs", len(filtered_df))
        with col2:
            st.metric("Blocked", filtered_df['blocked'].sum())
        with col3:
            st.metric("Avg Risk", f"{filtered_df['overall_risk_score'].mean():.1f}")
        with col4:
            st.metric("PII Detected", filtered_df['pii_detected'].sum())
    else:
        st.info("No logs match the selected criteria.")

def render_data_management():
    """Sample data management interface"""
    st.header("� Sample Data Management")
    
    tab1, tab2 = st.tabs(["Customer Orders", "Mitigation Policies"])
    
    with tab1:
        st.subheader("Customer Orders Database")
        
        conn = sqlite3.connect(DB_FILE)
        orders_df = pd.read_sql_query("SELECT * FROM customer_orders", conn)
        conn.close()
        
        st.dataframe(orders_df, use_container_width=True)
        
        # Add new order
        with st.expander("➕ Add Sample Order"):
            with st.form("add_order_form"):
                col1, col2 = st.columns(2)
                with col1:
                    order_id = st.text_input("Order ID")
                    email = st.text_input("Customer Email")
                    name = st.text_input("Customer Name")
                    product = st.text_input("Product")
                
                with col2:
                    status = st.selectbox("Status", ["Processing", "Shipped", "Delivered", "Cancelled"])
                    amount = st.number_input("Amount", min_value=0.0, format="%.2f")
                    eta = st.date_input("ETA")
                
                if st.form_submit_button("Add Order"):
                    conn = sqlite3.connect(DB_FILE)
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO customer_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (order_id, email, name, product, status, amount, 
                         datetime.date.today().isoformat(), eta.isoformat()))
                    conn.commit()
                    conn.close()
                    st.success("Order added successfully!")
                    st.rerun()
    
    with tab2:
        st.subheader("Risk Mitigation Policies")
        
        conn = sqlite3.connect(DB_FILE)
        policies_df = pd.read_sql_query("SELECT * FROM mitigation_policies", conn)
        conn.close()
        
        st.dataframe(policies_df, use_container_width=True)

def render_database_management():
    """Interactive chatbot: chat only (analysis moved to dashboard pages)"""
    st.header("🤖 AIRMS Chatbot")

    col1 = st.columns([1])[0]

    with col1:
        st.subheader("💬 Chat")

        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history (user + assistant only)
        for message in st.session_state.chat_history:
            if message.get('role') == 'user':
                st.markdown(f"**👤 You:** {message.get('content')}" )
            else:
                st.markdown(f"**🤖 AIRMS:** {message.get('content')}")

        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Enter your message:",
                height=100,
                placeholder="Ask your question"
            )
            col_a, col_b = st.columns(2)
            with col_a:
                submit_button = st.form_submit_button("Send Message 📤")
            with col_b:
                clear_button = st.form_submit_button("Clear Chat 🗑️")

        if clear_button:
            st.session_state.chat_history = []
            st.rerun()

        if submit_button and user_input:
            # Add user message
            st.session_state.chat_history.append({'role':'user','content':user_input})

            # Process through workflow and store full result in session state (not shown inline)
            with st.spinner("🔄 Processing..."):
                workflow_result = process_user_query(user_input)
            st.session_state.last_workflow = workflow_result

            # Append assistant content only (truncate to keep chat concise)
            assistant_text = workflow_result.final_response
            if len(assistant_text) > 800:
                assistant_text = assistant_text[:800].rsplit('\n', 1)[0] + '\n...'
            st.session_state.chat_history.append({'role':'assistant','content':assistant_text})
            st.rerun()

    # minimal chat UI: no session info or controls exposed here

def show_workflow_details(result: WorkflowResult):
    """Display detailed workflow analysis"""
    
    # Processing time and basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Processing Time", f"{result.processing_time:.3f}s")
    with col2:
        risk_color = {
            'CRITICAL': 'risk-critical',
            'HIGH': 'risk-high', 
            'MEDIUM': 'risk-medium',
            'LOW': 'risk-low'
        }.get(result.risk_assessment.risk_level, 'risk-low')
        st.markdown(f"**Risk Level:** <span class='{risk_color}'>{result.risk_assessment.risk_level}</span>", 
                   unsafe_allow_html=True)
    with col3:
        st.metric("Risk Score", f"{result.risk_assessment.overall_risk_score}/100")
    
    # Workflow steps
    st.markdown("**🔄 Workflow Steps:**")
    for i, log in enumerate(result.workflow_logs, 1):
        st.markdown(f"**{i}.** {log}")
    
    # Risk assessment details
    st.markdown("**🔍 Risk Assessment:**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"• **PII Detected:** {'Yes' if result.risk_assessment.pii_detected else 'No'}")
        st.write(f"• **Bias Score:** {result.risk_assessment.bias_fairness_score:.2f}")
    with col2:
        st.write(f"• **Adversarial:** {'Yes' if result.risk_assessment.adversarial_detected else 'No'}")
        st.write(f"• **Toxicity Score:** {result.risk_assessment.toxicity_score:.2f}")
    
    # Mitigation actions
    if result.risk_assessment.mitigation_actions:
        st.markdown("**🛡️ Mitigation Actions:**")
        for action in result.risk_assessment.mitigation_actions:
            st.write(f"• {action}")
    
            # Input comparison
            if result.original_input != result.sanitized_input:
                st.markdown("**🔄 Input Sanitization:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("Original", result.original_input, height=100, disabled=True, key=f"show_orig_{hash(result.original_input)}")
                with col2:
                    st.text_area("Sanitized", result.sanitized_input, height=100, disabled=True, key=f"show_san_{hash(result.sanitized_input)}")

def render_workflow_analytics():
    """Render comprehensive workflow analytics"""
    st.header("📊 Workflow Analytics Dashboard")
    
    conn = sqlite3.connect(DB_FILE)
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = pd.read_sql_query("SELECT COUNT(*) as count FROM workflow_logs", conn).iloc[0]['count']
        st.metric("Total Processed", total)
    
    with col2:
        blocked = pd.read_sql_query("SELECT COUNT(*) as count FROM workflow_logs WHERE blocked = 1", conn).iloc[0]['count']
        st.metric("Blocked Requests", blocked)
    
    with col3:
        avg_risk = pd.read_sql_query("SELECT AVG(overall_risk_score) as avg FROM workflow_logs", conn).iloc[0]['avg']
        st.metric("Avg Risk Score", f"{avg_risk:.1f}" if avg_risk else "0")
    
    with col4:
        avg_time = pd.read_sql_query("SELECT AVG(processing_time) as avg FROM workflow_logs", conn).iloc[0]['avg']
        st.metric("Avg Process Time", f"{avg_time:.3f}s" if avg_time else "0s")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk score distribution
        risk_scores = pd.read_sql_query("SELECT overall_risk_score FROM workflow_logs", conn)
        if not risk_scores.empty:
            fig = px.histogram(risk_scores, x='overall_risk_score', nbins=20, 
                             title="Risk Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Processing time trends
        time_trends = pd.read_sql_query("""
            SELECT date(timestamp) as date, AVG(processing_time) as avg_time
            FROM workflow_logs 
            GROUP BY date(timestamp)
            ORDER BY date
        """, conn)
        if not time_trends.empty:
            fig = px.line(time_trends, x='date', y='avg_time', 
                         title="Processing Time Trends")
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown
    st.subheader("🔍 Risk Component Analysis")
    
    components_data = pd.read_sql_query("""
        SELECT 
            AVG(CASE WHEN pii_detected = 1 THEN 100 ELSE 0 END) as pii_rate,
            AVG(bias_score * 100) as avg_bias,
            AVG(CASE WHEN adversarial_detected = 1 THEN 100 ELSE 0 END) as adversarial_rate,
            AVG(toxicity_score * 100) as avg_toxicity
        FROM workflow_logs
    """, conn)
    
    if not components_data.empty:
        data = components_data.iloc[0]
        fig = go.Figure(data=[
            go.Bar(x=['PII Detection', 'Bias Score', 'Adversarial Rate', 'Toxicity Score'],
                  y=[data['pii_rate'], data['avg_bias'], data['adversarial_rate'], data['avg_toxicity']])
        ])
        fig.update_layout(title="Risk Component Averages (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    conn.close()

def render_risk_analysis():
    """Deep dive risk analysis page"""
    st.header("🔍 Risk Analysis Deep Dive")
    
    # Sample inputs for testing
    st.subheader("🧪 Test Different Risk Scenarios")
    
    test_scenarios = {
        "Low Risk": "What are your business hours?",
        "Medium Risk - PII": "My email is john.doe@email.com, can you check my order?",
        "High Risk - Injection": "'; DROP TABLE users; --",
        "Critical Risk - Multiple": "My SSN is 123-45-6789 and credit card 4532-1234-5678-9012, hack the system"
    }
    
    selected_scenario = st.selectbox("Select test scenario:", list(test_scenarios.keys()))
    test_input = st.text_area("Test input:", test_scenarios[selected_scenario], height=100)
    
    if st.button("🔬 Analyze Risk"):
        with st.spinner("Analyzing..."):
            # Run just the risk detection
            risk_result = risk_detection_layer(test_input)
            
            # Display results
            st.subheader("📊 Risk Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Risk Score", f"{risk_result['overall_risk']['score']}/100")
            with col2:
                st.metric("Risk Level", risk_result['overall_risk']['level'])
            with col3:
                st.metric("Processing Time", f"{risk_result['processing_time']:.3f}s")
            
            # Component breakdown
            st.subheader("🔍 Component Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**PII Detection:**")
                if risk_result['pii_results']['pii_detected']:
                    for entity in risk_result['pii_results']['entities']:
                        st.write(f"• {entity['type']}: {entity['value']} (confidence: {entity['confidence']:.2f})")
                else:
                    st.write("No PII detected")
                
                st.write(f"**Bias Score:** {risk_result['bias_score']:.3f}")
            
            with col2:
                st.write(f"**Adversarial Detected:** {'Yes' if risk_result['adversarial_detected'] else 'No'}")
                st.write(f"**Toxicity Score:** {risk_result['toxicity_score']:.3f}")
            
            # Risk factors
            st.subheader("⚠️ Risk Factors Identified")
            for factor in risk_result['overall_risk']['factors']:
                st.write(f"• {factor}")
            
            # Sanitized output
            if risk_result['pii_results']['sanitized_text'] != test_input:
                st.subheader("🛡️ Sanitized Input")
                st.code(risk_result['pii_results']['sanitized_text'])

def render_workflow_logs():
    """Display privacy-focused workflow logs with risk analysis only"""
    st.header("📋 Privacy-Safe Workflow Logs")
    st.info("🔒 **Privacy First**: Only risk analysis data is stored and displayed - no user conversations")
    
    conn = sqlite3.connect(DB_FILE)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.selectbox("Risk Level:", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with col2:
        blocked_filter = st.selectbox("Status:", ["All", "Blocked", "Processed"])
    with col3:
        limit = st.number_input("Show last N records:", min_value=10, max_value=1000, value=50)
    
    # Build query
    query = "SELECT * FROM workflow_logs WHERE 1=1"
    if risk_filter != "All":
        query += f" AND risk_level = '{risk_filter}'"
    if blocked_filter == "Blocked":
        query += " AND blocked = 1"
    elif blocked_filter == "Processed":
        query += " AND blocked = 0"
    
    query += f" ORDER BY timestamp DESC LIMIT {limit}"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df.empty:
        st.write(f"📊 Showing {len(df)} risk analysis logs")
        
        for _, row in df.iterrows():
            # Extract risk types
            risk_types = []
            if row['pii_detected']:
                risk_types.append("PII")
            if row['bias_score'] > 0.3:
                risk_types.append("BIAS")
            if row['adversarial_detected']:
                risk_types.append("ADVERSARIAL")
            if row['toxicity_score'] > 0.3:
                risk_types.append("TOXICITY")
            
            risk_types_str = ", ".join(risk_types) if risk_types else "NONE"
            
            status_icon = "🚫" if row['blocked'] else "✅"
            status_class = "blocked" if row['blocked'] else "passed"
            
            # Create privacy-safe title
            title = f"{status_icon} {row['timestamp']} | {row['risk_level']} Risk (Score: {row['overall_risk_score']}) | Types: {risk_types_str}"
            
            with st.expander(title):
                st.markdown(f"<div class='workflow-step {status_class}'>", unsafe_allow_html=True)
                
                # Risk Analysis Section
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🛡️ Risk Analysis:**")
                    st.write(f"• **Level**: {row['risk_level']}")
                    st.write(f"• **Score**: {row['overall_risk_score']}/100")
                    st.write(f"• **Status**: {'Blocked' if row['blocked'] else 'Processed'}")
                
                with col2:
                    st.markdown("**🔍 Detection Details:**")
                    st.write(f"• **PII Detected**: {'Yes' if row['pii_detected'] else 'No'}")
                    st.write(f"• **Bias Score**: {row['bias_score']:.3f}")
                    st.write(f"• **Adversarial**: {'Yes' if row['adversarial_detected'] else 'No'}")
                    st.write(f"• **Toxicity Score**: {row['toxicity_score']:.3f}")
                
                with col3:
                    st.markdown("**⚙️ System Info:**")
                    st.write(f"• **Session**: {row['session_id'][:8]}...")
                    st.write(f"• **Database**: {'Enabled' if row.get('db_accessed', 0) else 'Disabled'}")
                    st.write(f"• **Processing Time**: {row['processing_time']:.3f}s")
                
                # Mitigation Actions
                if row['mitigation_actions']:
                    st.markdown("**🔧 Mitigation Actions Applied:**")
                    actions = row['mitigation_actions'].split('; ')
                    for action in actions:
                        if action.strip():
                            st.write(f"• {action}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("### 💬 **User Asked:**")
                st.markdown(f"> {row['user_input']}")
                
                # Risk Detection Section
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🛡️ **Risks Detected:**")
                    risk_details = []
                    if row['pii_detected']:
                        risk_details.append(f"🔒 **PII Found**: Yes")
                    if row['bias_score'] > 0.3:
                        risk_details.append(f"⚖️ **Bias Score**: {row['bias_score']:.2f}")
                    if row['adversarial_detected']:
                        risk_details.append(f"🎯 **Adversarial**: Detected")
                    if row['toxicity_score'] > 0.3:
                        risk_details.append(f"☠️ **Toxicity**: {row['toxicity_score']:.2f}")
                    if not risk_details:
                        risk_details.append("✅ **No significant risks**")
                    
                    for detail in risk_details:
                        st.markdown(f"• {detail}")
                
                with col2:
                    st.markdown("### ⚙️ **Processing Info:**")
                    st.markdown(f"• **Risk Level**: {row['risk_level']}")
                    st.markdown(f"• **Risk Score**: {row['overall_risk_score']}/100")
                    st.markdown(f"• **Processing Time**: {row['processing_time']:.2f}s")
                    st.markdown(f"• **Database Used**: {row.get('database_used', 'N/A')}")
                    st.markdown(f"• **Domain**: {row.get('domain_type', 'general')}")
                
                # LLM Response Section
                if row['llm_response'] and row['llm_response'] != 'BLOCKED_AT_MITIGATION':
                    st.markdown("### 🤖 **LLM Response:**")
                    st.markdown(f"> {row['llm_response']}")
                
                # Final Response Section  
                st.markdown("### 📤 **Final Response to User:**")
                if row['blocked']:
                    st.error(f"🚫 **BLOCKED**: {row['final_response']}")
                else:
                    st.success(f"✅ **DELIVERED**: {row['final_response']}")
                
                # Mitigation Actions
                if row['mitigation_actions']:
                    st.markdown("### 🛠️ **Mitigation Actions Taken:**")
                    actions = [action.strip() for action in row['mitigation_actions'].split(';') if action.strip()]
                    for action in actions:
                        st.markdown(f"• {action}")
                
                # Detailed Workflow Steps (collapsible)
                if row['workflow_steps']:
                    with st.expander("🔍 View Detailed Processing Steps"):
                        steps = str(row['workflow_steps']).split(' -> ')
                        for i, step in enumerate(steps, 1):
                            st.write(f"**{i}.** {step}")
                
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No workflow logs found matching the selected criteria.")

def render_data_management():
    """Sample data management interface"""
    st.header("📦 Sample Data Management")
    
    tab1, tab2 = st.tabs(["Customer Orders", "Mitigation Policies"])
    
    with tab1:
        st.subheader("Customer Orders Database")
        
        conn = sqlite3.connect(DB_FILE)
        orders_df = pd.read_sql_query("SELECT * FROM customer_orders", conn)
        conn.close()
        
        st.dataframe(orders_df, use_container_width=True)
        
        # Add new order
        with st.expander("➕ Add Sample Order"):
            with st.form("add_order_form"):
                col1, col2 = st.columns(2)
                with col1:
                    order_id = st.text_input("Order ID")
                    email = st.text_input("Customer Email")
                    name = st.text_input("Customer Name")
                    product = st.text_input("Product")
                
                with col2:
                    status = st.selectbox("Status", ["Processing", "Shipped", "Delivered", "Cancelled"])
                    amount = st.number_input("Amount", min_value=0.0, format="%.2f")
                    eta = st.date_input("ETA")
                
                if st.form_submit_button("Add Order"):
                    conn = sqlite3.connect(DB_FILE)
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO customer_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (order_id, email, name, product, status, amount, 
                         datetime.date.today().isoformat(), eta.isoformat()))
                    conn.commit()
                    conn.close()
                    st.success("Order added successfully!")
                    st.rerun()
    
    with tab2:
        st.subheader("Risk Mitigation Policies")
        
        conn = sqlite3.connect(DB_FILE)
        policies_df = pd.read_sql_query("SELECT * FROM mitigation_policies", conn)
        conn.close()
        
        st.dataframe(policies_df, use_container_width=True)

def render_database_management():
    """Universal Database Management Interface"""
    st.header("🗄️ Database Management")
    st.markdown("Configure databases and LLM prompts for different domains and data sources.")
    
    tab1, tab2, tab3 = st.tabs(["📊 Active Configuration", "🔧 Database Registry", "🤖 LLM Prompts"])
    
    with tab1:
        st.subheader("📋 Current Active Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🗄️ Active Database:**")
            active_db = get_active_database()
            if active_db:
                st.info(f"""
                **Name:** {active_db['db_name']}  
                **Path:** {active_db['db_path']}  
                **Domain:** {active_db['domain_type']}  
                **Table:** {active_db['table_name']}  
                **Description:** {active_db['description']}
                """)
            else:
                st.warning("❌ No active database configured!")
        
        with col2:
            st.markdown("**🤖 Active LLM Prompt:**")
            active_prompt = get_active_llm_prompt()
            st.info(f"""
            **Domain:** {active_prompt['domain_type']}  
            **Description:** {active_prompt.get('description', 'N/A')}
            """)
            
            if st.button("👀 View Full System Prompt"):
                st.text_area("System Prompt:", active_prompt['system_prompt'], height=200, disabled=True)
    
    with tab2:
        st.subheader("🗄️ Database Registry & Management")
        
        # Display all registered databases
        databases = get_all_databases()
        
        if databases:
            st.markdown("**🗂️ Registered Databases:**")
            for db in databases:
                db_id, db_name, db_path, domain_type, description, table_name, created_at, is_active = db
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    status_icon = "✅" if is_active else "⚪"
                    st.markdown(f"""
                    {status_icon} **{db_name}**  
                    📁 `{db_path}`  
                    🏷️ Domain: {domain_type} | Table: {table_name}  
                    📝 {description}
                    """)
                
                with col2:
                    if not is_active and st.button(f"Activate", key=f"activate_db_{db_id}"):
                        set_active_database(db_id)
                        st.success(f"✅ Activated: {db_name}")
                        st.rerun()
                
                with col3:
                    if st.button(f"🔍 Analyze", key=f"analyze_db_{db_id}"):
                        with st.spinner("Analyzing database structure..."):
                            structure = analyze_database_structure(db_path)
                            if 'error' in structure:
                                st.error(f"❌ Error: {structure['error']}")
                            else:
                                st.json(structure)
                
                st.divider()
        
        # Add new database
        with st.expander("➕ Add New Database"):
            st.markdown("**Upload or Connect to New Database:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_db_name = st.text_input("Database Name*", placeholder="My Customer DB")
                new_domain = st.selectbox("Domain Type*", [
                    "ecommerce", "healthcare", "finance", "education", "hr", "legal", "general"
                ])
                new_description = st.text_area("Description", placeholder="What data does this contain?")
            
            with col2:
                upload_method = st.radio("Connection Method:", ["Upload SQLite File", "Specify Path"])
                
                if upload_method == "Upload SQLite File":
                    uploaded_file = st.file_uploader("Upload SQLite (.db) file", type=['db', 'sqlite', 'sqlite3'])
                    new_db_path = None
                    if uploaded_file:
                        # Save uploaded file
                        new_db_path = f"uploaded_{uploaded_file.name}"
                        with open(new_db_path, "wb") as f:
                            f.write(uploaded_file.read())
                        st.success(f"📁 File saved as: {new_db_path}")
                else:
                    new_db_path = st.text_input("Database Path*", placeholder="path/to/database.db")
                
                new_table_name = st.text_input("Primary Table Name*", placeholder="customers")
            
            if st.button("➕ Add Database"):
                if new_db_name and new_db_path and new_domain and new_table_name:
                    try:
                        db_id = f"{new_domain}_{new_db_name.lower().replace(' ', '_')}"
                        add_custom_database(db_id, new_db_name, new_db_path, new_domain, new_description, new_table_name)
                        st.success(f"✅ Database '{new_db_name}' added successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error adding database: {str(e)}")
                else:
                    st.warning("⚠️ Please fill in all required fields (marked with *)")
    
    with tab3:
        st.subheader("🤖 LLM System Prompts Management")
        
        # Display all prompts
        prompts = get_all_llm_prompts()
        
        if prompts:
            st.markdown("**📝 Configured Prompts:**")
            for prompt in prompts:
                prompt_id, domain_type, system_prompt, description, created_at, is_active = prompt
                
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    status_icon = "✅" if is_active else "⚪"
                    st.markdown(f"""
                    {status_icon} **{domain_type.upper()}** - {description}  
                    📝 {system_prompt[:100]}...
                    """)
                
                with col2:
                    if not is_active and st.button(f"Activate", key=f"activate_prompt_{prompt_id}"):
                        set_active_llm_prompt(prompt_id)
                        st.success(f"✅ Activated: {domain_type}")
                        st.rerun()
                
                # Expandable full prompt view
                with st.expander(f"📄 View Full Prompt: {domain_type}"):
                    st.text_area(f"System prompt for {domain_type}:", system_prompt, height=200, disabled=True, key=f"view_prompt_{prompt_id}")
                
                st.divider()
        
        # Add new prompt
        with st.expander("➕ Create New LLM Prompt"):
            st.markdown("**Create Custom System Prompt:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_prompt_domain = st.selectbox("Domain Type*", [
                    "custom", "ecommerce", "healthcare", "finance", "education", "hr", "legal", "general"
                ], key="new_prompt_domain")
                new_prompt_desc = st.text_input("Description*", placeholder="What does this assistant do?")
            
            with col2:
                if new_prompt_domain == "custom":
                    custom_domain = st.text_input("Custom Domain Name*", placeholder="marketing")
                    final_domain = custom_domain
                else:
                    final_domain = new_prompt_domain
                    st.info(f"Domain: {new_prompt_domain}")
            
            new_system_prompt = st.text_area("System Prompt*", 
                placeholder="You are a helpful assistant specialized in...", 
                height=150)
            
            if st.button("➕ Create Prompt"):
                if final_domain and new_prompt_desc and new_system_prompt:
                    try:
                        prompt_id = f"{final_domain}_prompt_{int(time.time())}"
                        add_custom_llm_prompt(prompt_id, final_domain, new_system_prompt, new_prompt_desc)
                        st.success(f"✅ Prompt for '{final_domain}' created successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error creating prompt: {str(e)}")
                else:
                    st.warning("⚠️ Please fill in all required fields (marked with *)")
        
        # Quick test section
        with st.expander("🧪 Test Current Configuration"):
            st.markdown("**Test the current active database and prompt:**")
            test_query = st.text_input("Test Query:", placeholder="What is my order status for john.doe@email.com?")
            
            if st.button("🧪 Test Query"):
                if test_query:
                    with st.spinner("Testing configuration..."):
                        # Get active config
                        active_db = get_active_database()
                        active_prompt = get_active_llm_prompt()
                        
                        st.markdown("**🗄️ Database Response:**")
                        if active_db:
                            db_result = query_dynamic_database(f"SELECT * FROM {active_db['table_name']} LIMIT 3", active_db)
                            st.code(db_result)
                        else:
                            st.warning("No active database")
                        
                        st.markdown("**🤖 System Prompt Preview:**")
                        test_prompt = get_system_prompt(test_query, None)
                        st.text_area("Generated prompt:", test_prompt[:500] + "...", height=100, disabled=True)

def render_workflow_diagram():
    """Display workflow diagram explanation"""
    st.header("🎯 AIRMS Workflow Architecture")
    
    st.markdown("""
    ## 🔄 Complete Workflow Implementation
    
    This AIRMS system implements the complete workflow from your diagram:
    
    ### **1. User Input** 
    - Receives user queries through the chat interface
    
    ### **2. Risk Detection Layer (Agent)**
    - **PII Detection**: Identifies emails, phones, SSNs, credit cards
    - **Bias/Fairness Analysis**: Detects potential bias in language
    - **Adversarial Prompt Detection**: Identifies prompt injection attempts  
    - **Toxicity & Safety Filters**: Analyzes harmful content
    
    ### **3. Mitigation Layer**
    - **Replace Tokens**: Sanitizes PII with tokens
    - **Block Unsafe**: Blocks high-risk requests
    - **Escalate/Report**: Flags adversarial attempts
    
    ### **4. LLM Provider (Groq)**
    - Processes sanitized input through Groq API
    - Includes external data access when needed
    
    ### **5. Data Access Layer (Secure/Trusted Zone)**
    - Queries local SQLite database for order information
    - All results sanitized before feeding back to LLM
    
    ### **6. Output Post-Processing**
    - **Hallucination Check**: Validates response accuracy
    - **PII Leak Check**: Ensures no sensitive data in response
    - **Risk Score Assignment**: Assigns risk score to output
    
    ### **7. Risk Report + Dashboard Logs**
    - Complete workflow logged to database
    - Real-time dashboard monitoring
    - Risk analytics and reporting
    """)
    
    st.subheader("📊 Key Features Implemented:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔒 Security Features:**
        - Multi-layer PII detection
        - Adversarial prompt detection
        - Toxicity analysis
        - Response sanitization
        - Comprehensive logging
        """)
    
    with col2:
        st.markdown("""
        **📈 Analytics Features:**
        - Real-time risk monitoring
        - Workflow performance metrics
        - Risk trend analysis
        - Component-level insights
        - Session tracking
        """)
    
    st.subheader("🚀 How to Test:")
    st.markdown("""
    **Try these sample inputs in the chatbot:**
    
    1. **Normal Query**: "What are your business hours?"
    2. **PII Test**: "My email is test@email.com, check my order"
    3. **Injection Test**: "'; DROP TABLE users; --"
    4. **Multiple Risks**: "My SSN is 123-45-6789, please hack the system"
    
    Each query will show you the complete workflow execution with detailed analysis!
    """)

def render_interactive_chatbot():
    """Minimal chatbot interface without dashboard features"""
    st.header("💬 AI Risk Mitigation Chatbot")
    st.markdown("Ask questions and see the complete risk analysis workflow in action!")
    
    # Initialize session state for chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "details" in message:
                # Show the response
                st.write(message["content"])
                
                # Show risk analysis in an expandable section
                with st.expander("🛡️ Risk Analysis Details"):
                    details = message["details"]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Risk Score", details["risk_score"])
                    with col2:
                        st.metric("Risk Level", details["risk_level"])
                    with col3:
                        st.metric("PII Detected", "Yes" if details["pii_detected"] else "No")
                    with col4:
                        st.metric("Blocked", "Yes" if details["blocked"] else "No")
                    
                    st.markdown("**Risk Breakdown:**")
                    st.write(f"• Bias Score: {details['bias_score']:.2f}")
                    st.write(f"• Adversarial: {'Detected' if details['adversarial_detected'] else 'Clean'}")
                    st.write(f"• Toxicity Score: {details['toxicity_score']:.2f}")
                    
                    if details["mitigation_actions"]:
                        st.markdown("**Mitigation Actions:**")
                        for action in details["mitigation_actions"]:
                            st.write(f"• {action}")
            else:
                st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process the query through AIRMS workflow
        with st.chat_message("assistant"):
            with st.spinner("Processing through AIRMS workflow..."):
                try:
                    # Process the user query through the complete workflow
                    result = process_user_query(prompt)
                    
                    # Prepare response and details
                    response_content = result.final_response
                    risk_details = {
                        "risk_score": result.risk_assessment.overall_risk_score,
                        "risk_level": result.risk_assessment.risk_level,
                        "pii_detected": result.risk_assessment.pii_detected,
                        "bias_score": result.risk_assessment.bias_fairness_score,
                        "adversarial_detected": result.risk_assessment.adversarial_detected,
                        "toxicity_score": result.risk_assessment.toxicity_score,
                        "mitigation_actions": result.risk_assessment.mitigation_actions,
                        "blocked": result.blocked
                    }
                    
                    # Display the response
                    st.write(response_content)
                    
                    # Show risk analysis
                    with st.expander("🛡️ Risk Analysis Details", expanded=result.blocked):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Risk Score", risk_details["risk_score"])
                        with col2:
                            risk_color = {
                                'CRITICAL': '🔴',
                                'HIGH': '🟠', 
                                'MEDIUM': '🟡',
                                'LOW': '🟢'
                            }.get(risk_details["risk_level"], '⚪')
                            st.metric("Risk Level", f"{risk_color} {risk_details['risk_level']}")
                        with col3:
                            st.metric("PII Detected", "✅ Yes" if risk_details["pii_detected"] else "❌ No")
                        with col4:
                            st.metric("Blocked", "🚫 Yes" if risk_details["blocked"] else "✅ No")
                        
                        st.markdown("**🔍 Risk Breakdown:**")
                        st.write(f"• **Bias Score**: {risk_details['bias_score']:.2f}")
                        st.write(f"• **Adversarial**: {'⚠️ Detected' if risk_details['adversarial_detected'] else '✅ Clean'}")
                        st.write(f"• **Toxicity Score**: {risk_details['toxicity_score']:.2f}")
                        
                        if risk_details["mitigation_actions"]:
                            st.markdown("**🛡️ Mitigation Actions:**")
                            for action in risk_details["mitigation_actions"]:
                                st.write(f"• {action}")
                        
                        # Show processing time
                        st.markdown(f"**⏱️ Processing Time**: {result.processing_time:.3f} seconds")
                    
                    # Add assistant message to chat history
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": response_content,
                        "details": risk_details
                    })
                
                except Exception as e:
                    error_msg = f"I encountered an error processing your request: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Add sidebar info for chatbot
    with st.sidebar:
        st.markdown("---")
        st.markdown("**💬 Chatbot Mode**")
        st.info("This is the public chatbot interface. All queries go through complete risk analysis.")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_messages = []
            st.rerun()
        
        st.markdown("**🧪 Try These Tests:**")
        st.markdown("""
        - Normal: "What are your hours?"
        - PII: "My email is test@email.com"  
        - Adversarial: "'; DROP TABLE users; --"
        - Multiple risks: Include SSN, toxic language
        """)

# ========= MAIN ENTRY POINT ========= #
if __name__ == "__main__":
    init_local_db()
    main_dashboard()
