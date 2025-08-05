import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import pandas as pd
import os
import re
import random
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import openai
import threading

class ChatGPTClient:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.model = "gpt-3.5-turbo"
        self.temperature = 0.7
        self.max_tokens = 150
        
    def get_response(self, prompt: str, context: str = None) -> str:
        messages = []
        
        if context:
            messages.append({"role": "system", "content": context})
            
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            print(f"Error calling ChatGPT API: {str(e)}")
            return None

class BankingChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_title_bar()
        self.create_chat_interface()
        self.chatbot = AdvancedBankingChatbot("mykey")  # Using API key "mykey"
        self.display_welcome_message()
        
    def setup_window(self):
        self.root.title("Premier MyBank Assistant")
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
        
        tk.Label(header, text="MyBank Virtual Assistant by Sunil, Ibhrahim & Isaac", bg='#0056b3', 
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

class AdvancedBankingChatbot:
    def __init__(self, api_key: str = None):
        self.df = None
        self.vectorizer = None
        self.le = None
        self.load_banking777_data()
        self.initialize_services()
        
        # Initialize ChatGPT client if API key is provided
        self.chatgpt_client = None
        if api_key:
            try:
                self.chatgpt_client = ChatGPTClient(api_key)
                print("ChatGPT integration enabled")
            except Exception as e:
                print(f"Failed to initialize ChatGPT: {str(e)}")
    
    def load_banking777_data(self):
        try:
            dataset_paths = [
                'banking777.csv',
                'data/banking777.csv',
                'datasets/banking777.csv',
                os.path.join(os.path.dirname(__file__), 'banking777.csv'),
                os.path.join(os.path.dirname(__file__), 'data/banking777.csv')
            ]
            
            for path in dataset_paths:
                if os.path.exists(path):
                    self.df = pd.read_csv(path)
                    print(f"Successfully loaded Banking777 dataset from {path}")
                    break
            else:
                raise FileNotFoundError("Banking777 dataset not found")
            
            # Enhanced preprocessing
            self.df['text'] = self.df['text'].str.lower().str.replace('[^\w\s]', '')
            
            # Initialize vectorizer with better parameters
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),  # Consider single words and word pairs
                min_df=2  # Ignore terms that appear in fewer than 2 documents
            )
            self.X = self.vectorizer.fit_transform(self.df['text'])
            self.le = LabelEncoder()
            self.y = self.le.fit_transform(self.df['label'])
            
        except Exception as e:
            print(f"Error loading Banking777 dataset: {str(e)}")
            self.df = None
            self.vectorizer = None
            self.le = None

    def initialize_services(self):
        if self.df is not None:
            self.services = {}
            for label in self.df['label'].unique():
                self.services[label] = {
                    "patterns": [],  # Will be populated from dataset
                    "responses": [
                        f"I can help with {label.replace('_', ' ')}. Please provide more details.",
                        f"For {label.replace('_', ' ')}, please visit our website or contact customer service."
                    ]
                }
            
            # Extract common patterns from the dataset
            for idx, row in self.df.iterrows():
                label = row['label']
                text = row['text']
                # Add the first 3 words as a pattern
                pattern = ' '.join(text.split()[:3])
                if pattern and len(pattern) > 5:  # Only add meaningful patterns
                    if pattern not in self.services[label]["patterns"]:
                        self.services[label]["patterns"].append(pattern)
            
            # Enhance specific services with better responses
            enhanced_services = {
                "card_lost_or_stolen": {
                    "responses": [
                        "Please call our 24/7 helpline at 1-800-PRM-BANK to report immediately.",
                        "I can help you block your card temporarily through our secure verification process."
                    ],
                    "patterns": ["lost my card", "card stolen", "missing card"]
                },
                "transfer_money": {
                    "responses": [
                        "You can transfer funds between accounts using our online banking (daily limit: $5,000).",
                        "For money transfers, please authenticate first for security."
                    ],
                    "patterns": ["transfer money", "send money", "wire transfer"]
                },
                "account_balance": {
                    "responses": [
                        "Your current balance is available when you log in to your online account.",
                        "For security, I cannot disclose account balances here. Please check via our app."
                    ],
                    "patterns": ["check balance", "current balance", "account balance"]
                }
            }
            
            for service, data in enhanced_services.items():
                if service in self.services:
                    self.services[service]["responses"] = data["responses"]
                    self.services[service]["patterns"].extend(data["patterns"])
        else:
            self.services = {
                "greetings": {
                    "patterns": ["hello", "hi", "hey"],
                    "responses": [
                        "Hello! Welcome to Premier MyBank. How may I assist you?",
                        "Hi there! Thank you for contacting us."
                    ]
                },
                "balance": {
                    "patterns": ["balance", "how much"],
                    "responses": [
                        "You can check your balance online, via mobile app, or at an ATM.",
                        "For balance inquiries, please log in to your online banking."
                    ]
                },
                "transfer": {
                    "patterns": ["transfer", "send money"],
                    "responses": [
                        "Transfers can be made through online banking with a daily limit of $5,000.",
                        "For money transfers, please authenticate first."
                    ]
                },
                "bill": {
                    "patterns": ["pay bill", "make payment"],
                    "responses": [
                        "You can pay bills through the 'Payments' section in online banking.",
                        "Bill payments require authentication for security."
                    ]
                }
            }

    def get_ml_response(self, message):
        """Improved ML response with better intent matching"""
        if self.vectorizer is None:
            return None
            
        # Enhanced preprocessing
        message = re.sub(r'[^\w\s]', '', message.lower())
        
        input_vec = self.vectorizer.transform([message])
        similarities = cosine_similarity(input_vec, self.X)
        most_similar_idx = similarities.argmax()
        
        # Only return response if similarity score is high enough
        if similarities[0, most_similar_idx] > 0.6:  # Increased threshold
            predicted_label = self.le.inverse_transform([self.y[most_similar_idx]])[0]
            return random.choice(self.services[predicted_label]['responses'])
        return None

    def get_chatgpt_response(self, message: str) -> str:
        if not self.chatgpt_client:
            return None
            
        banking_context = """
        You are a helpful banking assistant for Premier MyBank. 
        Follow these rules:
        1. Be concise and professional
        2. Never ask for sensitive information
        3. For account-specific queries, direct to online banking
        4. Don't provide financial advice
        5. If unsure, say "Please contact customer service"
        """
        
        try:
            response = self.chatgpt_client.get_response(message, banking_context)
            if response:
                # Filter out any problematic responses
                blocked_phrases = [
                    "sorry, i can't",
                    "as an ai",
                    "i don't have access",
                    "i cannot provide"
                ]
                
                if not any(phrase in response.lower() for phrase in blocked_phrases):
                    return response
            return None
        except Exception as e:
            print(f"ChatGPT error: {str(e)}")
            return None

    def respond(self, message):
        original_message = message  # Keep original for ChatGPT
        message = message.lower()
        
        # 1. First check exact pattern matches in rule-based responses
        for service, data in self.services.items():
            if "patterns" in data:
                for pattern in data["patterns"]:
                    if re.search(r'\b' + re.escape(pattern) + r'\b', message):
                        return random.choice(data["responses"])
        
        # 2. Then try ML-based responses from Banking777 dataset
        if self.vectorizer is not None:
            ml_response = self.get_ml_response(message)
            if ml_response:
                return ml_response
        
        # 3. Finally try ChatGPT if enabled (using original message)
        if self.chatgpt_client:
            chatgpt_response = self.get_chatgpt_response(original_message)
            if chatgpt_response:
                return chatgpt_response
        
        # Default response
        default_responses = [
            "I'm not sure I understand. Could you rephrase your banking question?",
            "For banking assistance, you can ask about account balances, transfers, or bill payments."
        ]
        return random.choice(default_responses)

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