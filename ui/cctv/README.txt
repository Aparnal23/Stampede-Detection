CCTV zones from your photographs
==================================

1. Put image files in this folder (JPG, PNG, WebP, GIF, or SVG).

2. Edit cctv_zones.json — each entry is one zone:
   - name          — label shown in the dashboard
   - camera_id     — short ID (e.g. CCTV-01)
   - image         — filename of the normal / default view
   - image_abnormal — optional: different photo when the feed is "abnormal" (crowd simulation)
   - anchor_lat, anchor_lng — map coordinates for any future location logic (optional)

3. Restart the Flask app after changing files.

If cctv_zones.json is missing or invalid, the server will try to create one zone per image file in this folder (sorted by filename).
