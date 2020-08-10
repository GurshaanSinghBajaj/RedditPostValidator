
chrome.runtime.onMessage.addListener((msg, sender, response) => {
    // First, validate the message's structure.
    if ((msg.from === 'popup') && (msg.subject === 'DOMInfo')) {
      // Collect the necessary data. 
      // (For your specific requirements `document.querySelectorAll(...)`
      //  should be equivalent to jquery's `$(...)`.)
      var domInfo = {
        total: document.querySelectorAll('*').length,
        inputs: document.querySelectorAll('input').length,
        buttons: document.querySelectorAll('button').length,
      };
	  
	  head_line = "";
	  divs=document.querySelectorAll('div')
	  for(i=0;i<divs.length;i++)
	  {
		  c = divs[i].getAttributeNames();
		  for(j=0;j<c.length;j++)
		  {
			  if(c[j]=="data-test-id") if(divs[i].getAttribute("data-test-id")=="post-content")
			  {
				  pps = divs[i].querySelectorAll('p');
				  for(k=0;k<pps.length;k++)
				  {
					  head_line += pps[k].textContent;
				  }
				  i=divs.length;
				  break;
			  }
		  }
	  }
	  console.log("headline - "+head_line);
	  
	  coms = document.querySelectorAll('.Comment');
	  all_comments = [];
	  for(i=0;i<coms.length;i++)
	  {
		  paras = coms[i].querySelectorAll('p');
		  for(j=0;j<paras.length;j++)
		  {
			  var l = all_comments.push(paras[j].textContent);
		  }
	  }
	  
	  if(head_line.length > 500) head_line = head_line.slice(0,275);
	  console.log("head's length = "+head_line.length);
	  
	  var pdata = []
	  for(i=0;i<all_comments.length;i++)
	  {
		  var h = head_line
		  var comment = all_comments[i].slice(0,225);
		  var s = h + comment;
		  if(s.length > 500)
		  {
			  h = h.slice(0,275);
		  }
		  s = comment + h;
		  pdata.push([s,"agree"]);
	  }
	  
	  console.log(pdata.length);
	  pdata = JSON.stringify(pdata);
      $.ajax({
        url: 'http://127.0.0.1:5000/square/',
		contentType: 'application/json;charset=UTF-8',
        data: pdata,
        method: 'POST',
        async:'asynchronous',
        success: function(data) {
          //console.log(domInfo)
          response(data);
        }
      });

      return true;
      //response(domInfo);
      // Directly respond to the sender (popup), 
      // through the specified callback.
      
    }
  });