# Privacy Policy

This Plugin is a **model proxy**: it forwards requests from Dify to the **Higress AI Gateway Model API** endpoint you configure (“Endpoint”).

---

## What data the Plugin processes

To provide model capabilities, the Plugin may process and forward:
- **Model inputs** you send in Dify (e.g., prompts/messages/texts/documents, depending on the capability you use)
- **Model outputs** returned by your Endpoint (so Dify can display the result)
- **Configuration & credentials** you set in Dify for this Plugin (e.g., Endpoint URL, API Key / HMAC key & secret, extra headers)
- Optional **`user`** identifier passed by Dify to model APIs

---

## What the Plugin does NOT do

- **No database**: the Plugin does **not** store prompts/outputs in its own database.  
- **No analytics**: the Plugin does **not** perform tracking, analytics, or profiling.  
- **No fixed third parties**: the Plugin does **not** integrate with any fixed third-party service. Data is sent only to the **Endpoint you configure**.

---

## Data sharing

The Plugin sends your data to the **Endpoint you configure**. Your Endpoint (Higress) may route to upstream model services depending on **your** gateway configuration. Please review your own infrastructure and upstream providers’ policies if applicable.

---

## Storage & security

- Plugin configuration/credentials are managed and stored by **Dify** as part of plugin/model configuration.
- Requests are sent over HTTP(S) to your configured Endpoint.

