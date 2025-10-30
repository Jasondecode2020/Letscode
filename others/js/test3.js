const ws = new WebSocket('wss://x8wc4qccb8.execute-api.ap-southeast-2.amazonaws.com/production');

ws.onopen = () => {
  console.log('✅ Connected!');
  ws.send(JSON.stringify({ action: 'sendmessage', message: 'Hello!' }));
};

ws.onmessage = (event) => {
  console.log('📨 Received:', event.data);
};

ws.onerror = (error) => {
  console.error('❌ Error:', error);
};

ws.onclose = () => {
  console.log('🔌 Disconnected');
};