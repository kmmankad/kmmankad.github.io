---
layout: post
title: "Replace Text In Incoming Email with Outlook VBA"
date: 2016-02-09 21:38:07 +0530
comments: true
categories: [Outlook,VBA] 
---
After a quick and very helpful markdown tutorial over at [www.markdowntutorial.com](www.markdowntutorial.com), heres a post.

As someone who works for a large tech company, I need to write a lot of email where I need to refer to one or more bugs or commit-IDs, and manually adding hyperlinks to emails soon lost its charm. In an age where almost everything we encounter in our day that can be programmed now - surely there had to be a better way to do this. I had done some tinkering earlier with Excel VBA, but Outlook was something I never looked deeper into.

Some googling later, I chanced on this [superuser answer](http://superuser.com/a/464027), that seemed to document exactly what I wanted. Which was basically, I'd write an email with some text like this:
> Please pull the fix for Bug#123456

and I wanted a script to turn this into a hyperlink to a shortlink created from the Bug's ID# (123456), like so:
> Please pull the fix for [Bug#123456](http://coolbugs/123456)

So, I plonked the prescribed code in from the [superuser answer](http://superuser.com/a/464027) to see if it would work, and it didn't. While it did do the actual replace-text-with-a-hyperlink, it stripped the email of all formatting, and the hyperlink wasn't the kind I was expecting either (I wanted the link text to remain intact). Some more time reading about Outlook VBA and the MailItem class revealed that this wasn't the right way to tackle this when your Outlook Client uses the HTML Editor for composing and viewing email. So I decided to roll my own solution.

The code is available in this Github repo - [OutlookEmailHyperlinker](https://github.com/kmmankad/OutlookEmailHyperlinker) and works with Outlook 2013. I'm not sure if further description into its gory details is that attractive, so I'll keep it to this description on this for now.

Pull requests are most welcome!

### Code
* [OutlookEmailHyperlinker](https://github.com/kmmankad/OutlookEmailHyperlinker) 

### References
* [MSDN Reference on working with Outlook Email Bodies](https://msdn.microsoft.com/en-us/library/dd492012%28v=office.12%29.aspx)



