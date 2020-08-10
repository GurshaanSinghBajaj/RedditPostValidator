const setDOMInfo = info => {
    document.getElementById('summarize').style.display = 'none';
    document.getElementById('Agree').textContent = "Agree - "+info.total[0];
    document.getElementById('Discuss').textContent = "Discuss -  "+info.total[1];
    document.getElementById('Disagree').textContent = "Disagree - "+info.total[2];
    document.getElementById('Unrelated').textContent = "Unrelated - "+info.total[3];
    var sum=info.total[0]+info.total[1]+info.total[2]+info.total[3];
    var p0 = ((info.total[0])/(sum))*100;
    var p1 = ((info.total[1])/(sum))*100;
    var p2 = ((info.total[2])/(sum))*100;
    var p3 = ((info.total[3])/(sum))*100;
    var tc = "In this Post "+ p0 + "% comments agree ";
    tc += p1 + "% comments Discuss ";
    tc += p2 + "% comments Disagree with this Post and ";
    tc += p3 + "% are unrelated to the post";
    document.getElementById('percentage').textContent = tc;
  };
  
  function send_message()
  {
    chrome.tabs.query({
        active: true,
        currentWindow: true
      }, tabs => {
        // ...and send a request for the DOM info...
        chrome.tabs.sendMessage(
            tabs[0].id,
            {from: 'popup', subject: 'DOMInfo'},
            // ...also specifying a callback to be called 
            //    from the receiving end (content script).
            setDOMInfo);
      });
  }

  // Once the DOM is ready...
  window.addEventListener('DOMContentLoaded', () => {
    // ...query for the active tab...
    var but = document.getElementById('summarize');

but.addEventListener("click",send_message);
    
  });