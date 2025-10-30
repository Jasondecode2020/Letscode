const ws = new WebSocket('url');

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