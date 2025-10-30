(async function () {
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
  console.log(1)
  await sleep(1000)
  console.log(2)
  await sleep(1000)
  console.log(3)
  await sleep(1000)
  console.log(4)


  const current_date = new Date().toLocaleDateString('en-AU', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    timeZone: 'Australia/Brisbane'
  });
  console.log(current_date);
})()