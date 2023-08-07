const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const box = 25;
const canvasSize = 23;

// score variable
let score = 0;
//load snake starting postition
let snake = [];
snake[0] = {
  x: Math.floor((canvasSize / 2)) * box,
  y: Math.floor((canvasSize / 2)) * box
}

// direction getting pressed by arrow keys
let dir;
document.addEventListener('keydown', direction);

function direction(event) {
  if (event.keyCode == 37 && dir != 'RIGHT') {
    dir = 'LEFT';
  }
  if (event.keyCode == 38 && dir != 'DOWN') {
    dir = 'UP';
  }
  if (event.keyCode == 39 && dir != 'LEFT') {
    dir = 'RIGHT';
  }
  if (event.keyCode == 40 && dir != 'UP') {
    dir = 'DOWN';
  }
}

// set location of food, 1-22 number
let food = {
  x: Math.floor(1 + Math.random() * (canvasSize - 1)) * box,
  y: Math.floor(1 + Math.random() * (canvasSize - 1)) * box
}
// draw function
function draw() {
  // draw the background
  ctx.fillStyle = 'lightgreen';
  ctx.fillRect(box, box, canvasSize * box - box, canvasSize * box - box);
  // draw the snake head and tail
  for (let i = 0; i < snake.length; i++) {
    ctx.fillStyle = 'green';
    ctx.fillRect(snake[i].x, snake[i].y, box, box);
  }
  // move snake head
  let snakeX = snake[0].x;
  let snakeY = snake[0].y;

  if (dir == 'LEFT') {
    snakeX -= box;
  }
  if (dir == 'RIGHT') {
    snakeX += box;
  }
  if (dir == 'UP') {
    snakeY -= box;
  }
  if (dir == 'DOWN') {
    snakeY += box;
  }

  // if the snake eat the food
  if (snakeX == food.x && snakeY == food.y) {
    score += 1;
    food = {
      x: Math.floor(1 + Math.random() * (canvasSize - 1)) * box,
      y: Math.floor(1 + Math.random() * (canvasSize - 1)) * box
    }
  } else {
    snake.pop();
  }
  let newHead = {
    x: snakeX,
    y: snakeY
  }
  
  // check collision
  function collision(head, array) {
    for (let i = 0; i < array.length; i ++) {
      if (head.x == array[i].x && head.y == array[i].y) {
        return true;
      }
    }
    return false;
  }
  if (snakeX < box || snakeY < box || snakeX > (canvasSize - 1) * box || snakeY > (canvasSize - 1) * box
  || collision(newHead, snake)) {
    clearInterval(game);
  }
  snake.unshift(newHead);
  
  // draw in food
  ctx.fillStyle = 'red';
  ctx.fillRect(food.x, food.y, box, box);

  // draw score
  ctx.fillStyle = 'white';
  ctx.font = '24px Changa one';
  ctx.clearRect(0, 0, 50, 25);
  ctx.fillText(score, box, 0.8 * box)
  
}

let game = setInterval(draw, 200)