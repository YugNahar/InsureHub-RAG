//import React from 'react';
import AvaChat from './components/AvaChat';

function App() {
  return (
    <div style={{ 
      margin: 0, 
      padding: 0, 
      width: '100vw', 
      height: '100vh', 
      overflow: 'hidden',
      backgroundColor: '#0b1426'
    }}>
      <style>
        {`
          body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            background-color: #0b1426;
          }
          * {
            box-sizing: border-box;
          }
        `}
      </style>
      <AvaChat />
    </div>
  );
}

export default App;