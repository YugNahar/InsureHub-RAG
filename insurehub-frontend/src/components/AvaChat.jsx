import { useState, useRef, useEffect } from 'react';

const AvaChat = () => {
  // Generate a unique session ID once per page load
  // Generates a random 4-digit string (e.g., "4729") every time the page loads
  const [sessionId] = useState(() => Math.floor(1000 + Math.random() * 9000).toString());
  //const [sessionId] = useState(() => `session-${Math.floor(Math.random() * 1000000)}`);

  const [messages, setMessages] = useState([
    { 
      role: 'assistant', 
      content: "Hi! I'm Ava, your InsureHub Travel AI Agent ✈️. \n\nI can help you build your custom travel policy. Where are you traveling to, or do you have a passport/flight ticket you'd like to upload?" 
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Shared by both the text-input submit and quick-reply button taps —
  // tapping an option is just a faster way to send the exact same message a
  // typed reply would send, so both paths call this one function.
  const sendMessage = async (text) => {
    if (!text.trim()) return;

    const userMsg = { role: 'user', content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInputText('');
    setIsLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message: userMsg.content
        })
      });

      const data = await response.json();

      if (response.ok) {
        setMessages((prev) => [...prev, { role: 'assistant', content: data.response, options: data.options || null }]);
      } else {
        setMessages((prev) => [...prev, { role: 'assistant', content: "Sorry, I ran into a server error processing that message!" }]);
      }
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [...prev, { role: 'assistant', content: "Error connecting to the AI backend." }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    sendMessage(inputText);
  };

  // Tapping a quick-reply button sends its exact label as the next message —
  // same mechanic as typing it, just faster.
  const handleQuickReply = (optionText) => {
    if (isLoading || isUploading) return;
    sendMessage(optionText);
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsUploading(true);
    setMessages((prev) => [...prev, { role: 'user', content: `📎 Uploaded Document: ${file.name}` }]);

    const formData = new FormData();
    formData.append("session_id", sessionId);
    formData.append("file", file);

    try {
      // Local backend handles extraction (Protego) + deterministic field
      // merging + missing-field check + quote fetch, all in one call.
      const response = await fetch("http://127.0.0.1:8000/api/chat/upload-document", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setMessages((prev) => [...prev, { role: 'assistant', content: data.response, options: data.options || null }]);
      } else {
        setMessages((prev) => [...prev, { role: 'assistant', content: "The document scanner couldn't read that file. Could you try a clearer image?" }]);
      }
    } catch (error) {
      console.error("Upload error:", error);
      setMessages((prev) => [...prev, { role: 'assistant', content: "There was a network issue communicating with the extraction engine." }]);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', backgroundColor: '#0b1426', fontFamily: 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif', color: '#ffffff' }}>
      
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 24px', backgroundColor: '#0b1426', borderBottom: '1px solid #1e293b' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ position: 'relative' }}>
            <div style={{ width: '42px', height: '42px', borderRadius: '50%', backgroundColor: '#0f4c81', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '18px', fontWeight: '600', color: '#60a5fa' }}>A</div>
            <div style={{ position: 'absolute', bottom: '0', right: '0', width: '12px', height: '12px', borderRadius: '50%', backgroundColor: '#10b981', border: '2px solid #0b1426' }}></div>
          </div>
          <div>
            <h2 style={{ margin: '0 0 2px 0', fontSize: '17px', fontWeight: '600', letterSpacing: '0.3px' }}>Ava</h2>
            <span style={{ fontSize: '13px', color: '#94a3b8' }}>Travel AI Agent · online</span>
          </div>
        </div>
      </div>

      {/* Chat History */}
      <div style={{ flex: 1, padding: '24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
        {messages.map((msg, index) => (
          <div key={index} style={{ display: 'flex', flexDirection: 'column', gap: '8px', alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '75%' }}>
            <div style={{ display: 'flex', gap: '16px', alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
              {msg.role === 'assistant' && (
                <div style={{ width: '36px', height: '36px', borderRadius: '50%', backgroundColor: '#0f4c81', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '15px', fontWeight: '600', color: '#60a5fa', flexShrink: 0 }}>A</div>
              )}
              <div style={{
                backgroundColor: msg.role === 'user' ? '#00c3ff' : '#1e293b',
                color: msg.role === 'user' ? '#0a1128' : '#f8fafc',
                padding: '16px 20px',
                borderRadius: msg.role === 'user' ? '20px 20px 4px 20px' : '4px 20px 20px 20px',
                fontSize: '15px', lineHeight: '1.6', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', whiteSpace: 'pre-wrap'
              }}>
                {msg.content}
              </div>
            </div>

            {/* Quick-reply buttons — only shown on the LAST message so
                buttons from an earlier turn don't stay tappable forever
                once the conversation has moved on. */}
            {msg.role === 'assistant' && msg.options && index === messages.length - 1 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginLeft: '52px' }}>
                {msg.options.map((opt, optIndex) => (
                  <button
                    key={optIndex}
                    type="button"
                    onClick={() => handleQuickReply(opt)}
                    disabled={isLoading || isUploading}
                    style={{
                      backgroundColor: 'transparent',
                      color: '#00c3ff',
                      border: '1.5px solid #00c3ff',
                      borderRadius: '20px',
                      padding: '8px 18px',
                      fontSize: '14px',
                      fontWeight: '500',
                      cursor: (isLoading || isUploading) ? 'default' : 'pointer',
                      opacity: (isLoading || isUploading) ? 0.5 : 1,
                      transition: 'background-color 0.15s, color 0.15s',
                    }}
                    onMouseEnter={(e) => { if (!isLoading && !isUploading) { e.target.style.backgroundColor = '#00c3ff'; e.target.style.color = '#0a1128'; } }}
                    onMouseLeave={(e) => { e.target.style.backgroundColor = 'transparent'; e.target.style.color = '#00c3ff'; }}
                  >
                    {opt}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
        {(isLoading || isUploading) && (
          <div style={{ display: 'flex', gap: '16px', alignSelf: 'flex-start' }}>
            <div style={{ width: '36px', height: '36px', borderRadius: '50%', backgroundColor: '#0f4c81', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '15px', fontWeight: '600', color: '#60a5fa' }}>A</div>
            <div style={{ backgroundColor: '#1e293b', padding: '16px 20px', borderRadius: '4px 20px 20px 20px', fontSize: '15px', color: '#94a3b8' }}>
              {isUploading ? 'Extracting document data...' : 'Typing...'}
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div style={{ padding: '24px', backgroundColor: '#0b1426' }}>
        <form onSubmit={handleSendMessage} style={{ display: 'flex', alignItems: 'center', backgroundColor: '#0f172a', borderRadius: '30px', padding: '8px 16px', border: '1px solid #1e293b' }}>
          
          <input type="file" accept="image/*,application/pdf" ref={fileInputRef} style={{ display: 'none' }} onChange={handleFileUpload} />

          <button type="button" onClick={() => fileInputRef.current.click()} disabled={isLoading || isUploading}
            style={{ backgroundColor: 'transparent', color: '#94a3b8', border: 'none', cursor: (isLoading || isUploading) ? 'default' : 'pointer', padding: '8px', marginRight: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'color 0.2s' }} title="Upload Passport or Ticket" >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path></svg>
          </button>

          <input type="text" value={inputText} onChange={(e) => setInputText(e.target.value)} placeholder="Type a message..." disabled={isLoading || isUploading}
            style={{ flex: 1, backgroundColor: 'transparent', border: 'none', color: '#fff', outline: 'none', fontSize: '16px', padding: '8px' }} />

          <button type="submit" disabled={isLoading || isUploading || !inputText.trim()}
            style={{ backgroundColor: 'transparent', color: '#00c3ff', border: 'none', cursor: inputText.trim() ? 'pointer' : 'default', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '8px', opacity: inputText.trim() ? 1 : 0.4, transition: 'opacity 0.2s' }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
          </button>
        </form>
      </div>
    </div>
  );
};

export default AvaChat;