import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_brevo():
    # Brevo SMTP configuration - USE THESE CORRECT CREDENTIALS
    smtp_server = "smtp-relay.brevo.com"
    smtp_port = 587
    smtp_username = "9b1b80001@smtp-brevo.com"  # Your SMTP username from Brevo
    smtp_password = "xsmtpsib-168a51b51d3cb845422896971d05de2270ca09281098242edc65412e60e445de-yCSsIuskKzbs5Iar"  # Your SMTP key starting with xsmtpsib-
    
    # Email configuration
    sender_email = "projects.srinath@gmail.com"  # Your verified sender email
    sender_name = "Srinath"
    receiver_email = "srinathnulidonda@gmail.com"
    receiver_name = "Srinath N"
    
    # Create message
    message = MIMEMultipart("alternative")
    message["Subject"] = "Test Email from Brevo - Working!"
    message["From"] = f"{sender_name} <{sender_email}>"
    message["To"] = f"{receiver_name} <{receiver_email}>"
    
    # Email body
    text = """\
    Hi,
    
    This is a test email sent successfully via Brevo SMTP!
    
    Best regards,
    Srinath
    """
    
    html = """\
    <html>
      <body style="font-family: Arial, sans-serif;">
        <h2 style="color: #333;">Hello! 👋</h2>
        <p>This is a <b>successful test email</b> sent via Brevo SMTP.</p>
        <p style="color: #666;">
          Your SMTP configuration is working perfectly!
        </p>
        <p>Best regards,<br>
           <strong>Srinath</strong>
        </p>
      </body>
    </html>
    """
    
    # Convert to MIMEText objects
    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")
    
    # Attach parts
    message.attach(part1)
    message.attach(part2)
    
    try:
        # Create SMTP session
        print("Connecting to Brevo SMTP...")
        server = smtplib.SMTP(smtp_server, smtp_port)
        
        print("Starting TLS encryption...")
        server.starttls()
        
        print("Authenticating...")
        server.login(smtp_username, smtp_password)
        
        print("Sending email...")
        server.send_message(message)
        server.quit()
        
        print("✅ Email sent successfully!")
        print(f"   From: {sender_email}")
        print(f"   To: {receiver_email}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# Send the email
send_email_brevo()