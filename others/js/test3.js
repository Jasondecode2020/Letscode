// const ws = new WebSocket('url');

// ws.onopen = () => {
//   console.log('✅ Connected!');
//   ws.send(JSON.stringify({ action: 'sendmessage', message: 'Hello!' }));
// };

// ws.onmessage = (event) => {
//   console.log('📨 Received:', event.data);
// };

// ws.onerror = (error) => {
//   console.error('❌ Error:', error);
// };

// ws.onclose = () => {
//   console.log('🔌 Disconnected');
// };

const removeDuplicate = (arr) => {
  const s = new Set();
  return arr.map((item) => {
    if (s.has(item)) return 0;
    s.add(item);
    return item;
  })
}

const arr1 = [1, 1, 2, 3, 3];
const arr = removeDuplicate(arr1);
console.log(arr);