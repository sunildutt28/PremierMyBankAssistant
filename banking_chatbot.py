import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import pandas as pd
import os
import re
import random
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI  # Updated import for new API version
import threading

class ChatGPTClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)  # New client initialization
        self.model = "gpt-3.5-turbo"
        self.temperature = 0.3
        self.max_tokens = 100
        
    def get_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(  # Updated API call
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": """You are a banking assistant. Only answer banking-related questions. 
                    For account-specific questions, direct users to official channels."""
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()  # Updated attribute access
        except Exception as e:
            print(f"ChatGPT Error: {e}")
            return None

class AdvancedBankingChatbot:
    def __init__(self, api_key: str = None):
        self.df = None
        self.vectorizer = None
        self.le = None
        self.load_banking777_data()
        self.initialize_services()
        
        # Initialize ChatGPT only if API key is provided and valid
        self.chatgpt_client = ChatGPTClient(api_key) if api_key else None
        self.chatgpt_enabled = bool(api_key)
        
    def load_banking777_data(self):
        try:
            # Try to locate the dataset
            base_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join(base_dir, 'banking777.csv'),
                os.path.join(base_dir, 'data', 'banking777.csv'),
                os.path.join(base_dir, 'datasets', 'banking777.csv')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.df = pd.read_csv(path)
                    print(f"Loaded Banking777 dataset from: {path}")
                    
                    # Preprocess data
                    self.df['cleaned_text'] = self.df['text'].apply(
                        lambda x: re.sub(r'[^\w\s]', '', x.lower())
                    )
                    
                    # Initialize vectorizer with better parameters
                    self.vectorizer = TfidfVectorizer(
                        stop_words='english',
                        ngram_range=(1, 2),
                        min_df=2
                    )
                    self.X = self.vectorizer.fit_transform(self.df['cleaned_text'])
                    self.le = LabelEncoder()
                    self.y = self.le.fit_transform(self.df['label'])
                    break
                    
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.df = None

    def initialize_services(self):
        """Initialize with both dataset and hardcoded banking responses"""
        self.services = {
            # Greetings
            "greeting": {
                "patterns": ["hello", "hi", "hey", "good morning", "good afternoon"],
                "responses": [
                    "Hello! Welcome to Premier MyBank. How may I assist you today?",
                    "Hi there! Thank you for contacting Premier Bank."
                ]
            },
            # Banking services
            "balance_inquiry": {
                "patterns": ["balance", "how much", "account balance", "check balance"],
                "responses": [
                    "You can check your balance through our mobile app or online banking.",
                    "For account balance, please log in to your online banking account."
                ]
            },
            "money_transfer": {
                "patterns": ["transfer", "send money", "wire transfer", "move money"],
                "responses": [
                    "You can transfer money through our online banking or mobile app.",
                    "For money transfers, please authenticate through our secure banking portal."
                ]
            },
            # Add more services as needed
        }
        
        # Enhance with dataset if available
        if self.df is not None:
            for label in self.df['label'].unique():
                if label not in self.services:
                    self.services[label] = {
                        "patterns": [],
                        "responses": [
                            f"I can help with {label.replace('_', ' ')}.",
                            f"For {label.replace('_', ' ')}, please contact customer service."
                        ]
                    }

    def get_rule_based_response(self, message: str) -> str:
        """Check for exact pattern matches with priority"""
        message = re.sub(r'[^\w\s]', '', message.lower())
        
        for intent, data in self.services.items():
            for pattern in data.get("patterns", []):
                if re.search(r'\b' + re.escape(pattern) + r'\b', message):
                    return random.choice(data["responses"])
        return None

    def get_ml_response(self, message: str) -> str:
        """Use ML only if we have a good match"""
        if not self.vectorizer:
            return None
            
        message_clean = re.sub(r'[^\w\s]', '', message.lower())
        vec = self.vectorizer.transform([message_clean])
        similarities = cosine_similarity(vec, self.X)
        best_match_idx = similarities.argmax()
        
        if similarities[0, best_match_idx] > 0.65:  # High confidence threshold
            intent = self.le.inverse_transform([self.y[best_match_idx]])[0]
            return random.choice(self.services.get(intent, {}).get("responses", []))
        return None

    def respond(self, message: str) -> str:
        # 1. Try rule-based first
        if response := self.get_rule_based_response(message):
            return response
            
        # 2. Try ML-based if we have the dataset
        if response := self.get_ml_response(message):
            return response
            
        # 3. Only use ChatGPT if enabled and for clearly non-banking questions
        if self.chatgpt_enabled and not self.is_banking_question(message):
            if response := self.chatgpt_client.get_response(message):
                return response
                
        # Default banking response
        return ("I can help with banking services like account balances, transfers, "
               "and bill payments. Could you clarify your request?")

    def is_banking_question(self, message: str) -> bool:
        """Check if question is related to banking"""
        banking_keywords = [
            'account', 'balance', 'transfer', 'bank', 'loan', 
            'card', 'payment', 'bill', 'money', 'deposit'
        ]
        message = message.lower()
        return any(keyword in message for keyword in banking_keywords)

class BankingChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_title_bar()
        self.create_chat_interface()
        self.chatbot = AdvancedBankingChatbot("mykey") 
        self.display_welcome_message()
        
    def setup_window(self):
        self.root.title("MyBank Assistant - by Sunil Dutt, Ibrahim & Ishaku")
        self.root.geometry("700x600")
        self.root.minsize(600, 600)
        self.root.configure(bg='#f5f7fa')
        
        try:
            self.root.iconbitmap('assets/icon.ico')
        except Exception as e:
            print(f"Could not load window icon: {e}")

    def create_title_bar(self):
        self.title_bar = tk.Frame(self.root, bg='#003366', relief='raised', bd=0)
        self.title_bar.pack(fill='x')
        
        try:
            logo_img = Image.open('assets/logo.png')
            logo_img = logo_img.resize((120, 30), Image.Resampling.LANCZOS)
            self.logo = ImageTk.PhotoImage(logo_img)
            tk.Label(self.title_bar, image=self.logo, bg='#003366').pack(side='left', padx=10)
        except Exception as e:
            print(f"Could not load logo: {e}")
            tk.Label(self.title_bar, text="Premier MyBank", 
                    bg='#003366', fg='white', font=('Roboto', 10, 'bold')).pack(side='left', padx=10)
        
        controls = tk.Frame(self.title_bar, bg='#003366')
        controls.pack(side='right')
        
        tk.Button(controls, text='─', bg='#003366', fg='white', bd=0,
                 command=lambda: self.root.state('iconic'), font=('Segoe UI', 12)
                 ).pack(side='left', padx=5)
        
        tk.Button(controls, text='□', bg='#003366', fg='white', bd=0,
                 command=self.toggle_maximize, font=('Segoe UI', 10)
                 ).pack(side='left', padx=5)
        
        tk.Button(controls, text='✕', bg='#003366', fg='white', bd=0,
                 command=self.root.destroy, font=('Segoe UI', 10)
                 ).pack(side='left', padx=5)

    def create_chat_interface(self):
        self.chat_frame = tk.Frame(self.root, bg='white', bd=0)
        self.chat_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        header = tk.Frame(self.chat_frame, bg='#0056b3', height=50)
        header.pack(fill='x')
        
        tk.Label(header, text="MyBank Virtual Assistant", bg='#0056b3', 
                fg='white', font=('Roboto', 12, 'bold')).pack(side='left', padx=15)
        
        self.status = tk.Label(header, text="● Online", bg='#0056b3', 
                             fg='#4caf50', font=('Roboto', 10))
        self.status.pack(side='right', padx=15)
        
        self.chat_area = scrolledtext.ScrolledText(
            self.chat_frame, wrap=tk.WORD, width=60, height=20,
            bg='#f5f7fa', bd=0, highlightthickness=0,
            font=('Roboto', 10), state='disabled'
        )
        self.chat_area.pack(fill='both', expand=True, padx=0, pady=0)
        
        self.chat_area.tag_config('user', foreground='#333333', 
                                background='#e3f2fd', lmargin1=20, 
                                lmargin2=20, rmargin=20, 
                                spacing1=5, spacing3=5,
                                relief='solid', borderwidth=0)
        self.chat_area.tag_config('bot', foreground='#333333', 
                                 background='white', lmargin1=20, 
                                 lmargin2=20, rmargin=20,
                                 spacing1=5, spacing3=5,
                                 relief='solid', borderwidth=0)
        self.chat_area.tag_config('time', foreground='#777777', 
                                font=('Roboto', 8))
        
        input_frame = tk.Frame(self.chat_frame, bg='white')
        input_frame.pack(fill='x', pady=(0, 15))
        
        self.user_input = ttk.Entry(
            input_frame, font=('Roboto', 12)
        )
        self.user_input.pack(side='left', fill='x', expand=True, padx=(15, 5))
        self.user_input.bind('<Return>', self.send_message)
        
        send_btn = ttk.Button(
            input_frame, text="Send", 
            command=self.send_message
        )
        send_btn.pack(side='right', padx=(5, 15))
        
        quick_actions = tk.Frame(self.chat_frame, bg='white')
        quick_actions.pack(fill='x', padx=15, pady=(0, 15))
        
        actions = [
            ("Check Balance", "balance"),
            ("Transfer Money", "transfer"),
            ("Pay Bill", "bill"),
            ("Branch Info", "branch")
        ]
        
        for text, action in actions:
            ttk.Button(
                quick_actions, text=text, 
                command=lambda a=action: self.quick_action(a),
                width=15
            ).pack(side='left', expand=True, padx=5)

    def toggle_maximize(self):
        if self.root.state() == 'zoomed':
            self.root.state('normal')
        else:
            self.root.state('zoomed')

    def display_message(self, sender, message):
        timestamp = datetime.now().strftime("%H:%M")
        
        self.chat_area.config(state='normal')
        self.chat_area.insert('end', f"\n{message}\n", sender)
        self.chat_area.insert('end', f"{timestamp}\n", 'time')
        self.chat_area.config(state='disabled')
        self.chat_area.see('end')

    def display_welcome_message(self):
        welcome_msg = (
            "Welcome to Premier MyBank Assistant!\n\n"
            "I can help you with:\n"
            "- Account balances\n"
            "- Money transfers\n"
            "- Bill payments\n"
            "- Card services\n"
            "- Branch information\n\n"
            "How may I assist you today?"
        )
        self.display_message("bot", welcome_msg)

    def send_message(self, event=None):
        message = self.user_input.get().strip()
        if not message:
            return
            
        self.display_message("user", message)
        self.user_input.delete(0, 'end')
        
        threading.Thread(
            target=self.process_user_message,
            args=(message,),
            daemon=True
        ).start()

    def process_user_message(self, message):
        try:
            response = self.chatbot.respond(message)
            if not response:
                response = "I didn't understand that. Could you rephrase your banking question?"
            self.root.after(0, self.display_message, "bot", response)
        except Exception as e:
            error_msg = "Our systems are currently unavailable. Please try again later."
            if "exceeded" in str(e).lower():
                error_msg = "I'm getting too many requests. Please wait a moment and try again."
            self.root.after(0, self.display_message, "bot", error_msg)
            print(f"Error processing message: {str(e)}")

    def quick_action(self, action):
        actions = {
            "balance": "How do I check my account balance?",
            "transfer": "I want to transfer money to another account",
            "bill": "How can I pay my credit card bill?",
            "branch": "Where is the nearest branch location?"
        }
        self.user_input.delete(0, 'end')
        self.user_input.insert(0, actions[action])
        self.send_message()

if __name__ == "__main__":
    root = tk.Tk()
    
    try:
        from tkinter import font
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(family="Roboto", size=10)
    except:
        pass
    
    app = BankingChatbotGUI(root)
    root.mainloop()