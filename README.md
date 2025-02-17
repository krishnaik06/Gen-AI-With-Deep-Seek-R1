# Gen-AI-With-Deep-Seek-R1

A powerful local AI development environment featuring DeepSeek models for document analysis and code assistance.

## 🌟 Key Features

### 📚 DocuMind AI (rag_deep.py)
- PDF document analysis with chat interface
- Local processing for data privacy
- Configurable model sizes (1.5B/7B)
- Real-time progress tracking
- Persistent chat history
- Thought process visualization
- Chunk size optimization

### 💻 Code Companion (app.py)
- AI pair programming assistant
- Python code expertise
- Real-time debugging support
- Solution design assistance
- Continuous chat context

## 🚀 Quick Start

### Prerequisites
1. Python 3.8+
2. [Ollama](https://ollama.ai/) installed
3. DeepSeek models:
```bash
ollama pull deepseek-r1:1.5b
ollama pull deepseek-r1:7b
```

### Installation
1. Clone and setup:
```bash
git clone [your-repo-url]
cd Gen-AI-With-Deep-Seek-R1
pip install -r requirements.txt
```

2. Start Ollama:
```bash
ollama serve
```

3. Run applications:
```bash
# For document analysis:
streamlit run rag_deep.py

# For code assistance:
streamlit run app.py
```

## 💡 Usage Tips

### DocuMind AI
- Use 1.5B model for faster responses
- 7B model for complex documents
- Adjust chunk size for performance
- Reset button clears chat history

### Code Companion
- Clear questions get better answers
- Include context in code questions
- Use system prompts effectively

## 🔧 Troubleshooting

1. Model Issues
   - Verify Ollama is running
   - Check model availability: `ollama list`
   - Ensure port 11434 is accessible

2. Performance
   - Reduce chunk size for large documents
   - Use 1.5B model for faster responses
   - Monitor system resources

## 📚 Learn More
- [Ollama Documentation](https://ollama.ai/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [LangChain Documentation](https://python.langchain.com/docs)

## 🤝 Contributing
Contributions welcome! Please read our contributing guidelines.

## 📄 License
