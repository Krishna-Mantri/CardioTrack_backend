from supabase import create_client, Client

# ⛳ Replace these with your actual Supabase credentials
SUPABASE_URL="https://ojucakyzqgosmbdioxzz.supabase.co"
SUPABASE_KEY="sb_publishable_ANht-1HQDq19av1XA2WsmQ_pv7IXaAF"
# SUPABASE_URL= "https://your-supabase-url.supabase.co"
# SUPABASE_KEY= "your-supabase-key"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)



