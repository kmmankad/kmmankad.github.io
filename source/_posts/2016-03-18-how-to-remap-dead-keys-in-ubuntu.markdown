---
layout: post
title: "How to remap dead keys in Ubuntu"
date: 2016-03-18 19:14:32 +0530
comments: true
categories: 
---

My personal computer is an old Lenovo G560, whose keyboard recently started showing signs of wear, with the double-quote/single-quote key being the first to give out. As someone who likes to write code, losing the quote key is a real inconvenience. Sure, I could spend some cash for a new keyboard, but what good is your knowledge in software if it can't help workaround hardware problems? :P

So I decided to implement the lost key by re-mapping an existing key to behave like it. I chose the semicolon/colon key here. The idea is simple: I'd remap the semicolon/colon key to print single and double quotes with key modifiers (Right Alt and Shift) 

Alt + Semicolon = Single Quote
<img src="https://www.wpclipart.com/computer/keyboard_keys/computer_key_Alt.png" width="50"> + <img src="https://www.wpclipart.com/computer/keyboard_keys/computer_key_Colon_Semicolon.png" width="50"> = <img src="https://www.wpclipart.com/computer/keyboard_keys/computer_key_Quotation_Marks.png" width="50">

..and, Shift + Alt + Semicolon = Double Quote
<img src="https://www.wpclipart.com/computer/keyboard_keys/large_keys/computer_key_Shift.png" width="120"> + <img src="https://www.wpclipart.com/computer/keyboard_keys/computer_key_Alt.png" width="50"> + <img src="https://www.wpclipart.com/computer/keyboard_keys/computer_key_Colon_Semicolon.png" width="50"> = <img src="https://www.wpclipart.com/computer/keyboard_keys/large_keys/computer_key_Shift.png" width="120"> + <img src="https://www.wpclipart.com/computer/keyboard_keys/computer_key_Quotation_Marks.png" width="50">

To achieve this, what you need to do is create a custom keyboard layout. In Ubuntu, all the keyboard layouts (for different languages and such) are located at `/usr/share/X11/xkb/symbols/`. Now, you can chose to create a new file altogether, or add a variant to an existing file. I chose the latter, so I added the following lines to the US Keyboard layout located at `/usr/share/X11/xkb/symbols/us`

```
partial alphanumeric_keys
xkb_symbols "kmmankad" {
     name[Group1] = "English (US, with apostrophe|quotedbl mapped to colon|semicolon";
 
     include "us(basic)"
     key <AC10> { [ semicolon,colon, apostrophe, quotedbl ] };
 
     include "level3(ralt_switch)"
};
```
Here, I define a new keyboard layout named `kmmankad`, and its description is specified under `name[Group1]`. The `include "us(basic)"` line includes the basic US keyboard layout, since ours is really just one key away from the default US basic. Thats what the next line is about.
```
     key <AC10> { [ semicolon,colon, apostrophe, quotedbl ] };
```
`AC10` - A (for alphanumeric). The second letter could take values from A-E (rows 1 to 5, _bottom to top_ - rowA is the spacebar row) and 1-12 (key position in the row, going _left to right_). So with that co-ordinate system, we locate semicolon as ```AC10```. After the key is defined, you the next four values will correspond with what the key will produce by itself, with Shift, with AltGr (usually the right Alt key), and Shift+AltGr respectively. Easy peasy.

Now that we have this defined, We need to update ```/usr/share/X11/xkb/rules/evdev.xml``` to include this new layout.

```xml
        <variant>
          <configItem>
            <name>kmmankad</name>
	    <description>English (US, with apostrophe|quotedbl mapped to colon|semicolon)</description>
          </configItem>
        </variant>
```
Note that the name and description should match what you entered earlier in `/usr/share/X11/xkb/symbols/us`.

Now, restart your machine and you should have your new keyboard layout available under `System Settings` -> `Keyboard` -> `Text Entry` (see bottom right) -> `+`

Select, Test and Done. Hope this was helpful!
