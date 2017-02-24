---
layout: page
title: ""
permalink: /contact/
---
## Contact

<form action="https://getsimpleform.com/messages?form_api_token=38951d380d1b9e552d3ad20bb69eaadb" method="post">
  <!-- the redirect_to is optional, the form will redirect to the referrer on submission -->
  <input type='hidden' name='redirect_to' value='https://yuzukixx.github.io/contact_received' />
  <!-- all your input fields here.... -->
  <div> <p>Name</p>
  <p><input name = "name" size = "40" type="text" style="font-family:monospace;" /></p>
</div>
  <div> <p>Email Address<font color="red">* </font></p>
  <p><input name = "email" size = "40" type="text" required = "required" style="font-family:monospace;" /></p>
</div>

<div>
  <p>Message</p>
  <p><textarea name='message' cols = "80" rows ="10" resize = "none"></textarea></p>
</div>

<div>
	<p><input type = "checkbox" name="shouldContact" /> I would like a reply. </p>
</div>


  <input type='submit' value='Submit' style="font-family:calibri;font-size:30;" />
</form>