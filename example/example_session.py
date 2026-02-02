"""Example: Simple session memory evaluation and adaptive update using interfaces."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plugin.memory_adapter import MemoryAdapter
from plugin.memory_challenger import MemoryChallenger
from plugin.memory_evaluator import MemoryEvaluator, PASS_RATE_THRESHOLD
from plugin.memory_system import MemorySystem
from simple_adapter import SimpleMemoryAdapter
from simple_memory_system import SimpleMemorySystem

__all__ = ["process_session", "main", "EXAMPLE_SESSION_DATA"]


EXAMPLE_SESSION_DATA = {
    "user_id": "user_001",
    "session_id": "session_2024_01_15_001",
    "session_dialogue": """user: Hi, I'm planning a complex trip to Japan with multiple friends. Sarah, my college roommate, and I are leaving from New York on March 15th, 2024 at 2:30 PM on JAL Flight 004. We'll arrive in Tokyo on March 16th at 4:45 PM local time.
assistant: That sounds like an exciting adventure! A long flight from New York. Are you planning to stay in Tokyo first?
user: Yes, we're staying at the Park Hyatt Tokyo in Shinjuku for exactly 3 nights, checking out on March 19th. Sarah has been to Tokyo twice before in 2019 and 2022, but this is my first time visiting Japan.
assistant: Park Hyatt is a great choice with amazing views. Since Sarah has experience, she can help you navigate. What are your main interests for this trip?
user: I'm really passionate about traditional Japanese architecture and gardens. I want to visit at least 5 temples. Sarah is obsessed with anime and manga - she specifically wants to spend a full day in Akihabara on March 17th, which is a Sunday. We both love food and want to try authentic ramen, sushi, and kaiseki meals.
assistant: That's a wonderful mix! Kyoto would be perfect for temples. For Akihabara on Sunday, it'll be very busy but that's part of the experience. For food, I highly recommend Ichiran Ramen in Shibuya for ramen, and Sukiyabashi Jiro in Ginza for sushi - though you need reservations 3 months in advance. For kaiseki, try Kikunoi in Kyoto.
user: Thanks for the recommendations! We're planning to take the Shinkansen from Tokyo to Kyoto on March 19th at 9:00 AM. We've already booked Green Car seats 3A and 3B. In Kyoto, we're staying at Hiiragiya Ryokan for 4 nights, checking in on March 19th and out on March 23rd.
assistant: Hiiragiya is one of the oldest ryokans, established in 1818! The Green Car on Shinkansen is comfortable. Have you thought about what you'll do in Kyoto?
user: Yes, I've made a detailed itinerary. On March 20th, we'll visit Kiyomizu-dera Temple in the morning, then Fushimi Inari Shrine in the afternoon. On March 21st, we're going to Arashiyama Bamboo Grove early at 7 AM to avoid crowds, then the Golden Pavilion (Kinkaku-ji) at 2 PM. Sarah wants to visit the Ghibli Museum, but I told her it's in Mitaka, near Tokyo, not Kyoto.
assistant: Good planning! Early morning at Arashiyama is smart. The Ghibli Museum is indeed in Mitaka, about 30 minutes from central Tokyo. You'd need to go back or visit it before leaving Tokyo. Tickets are released on the 10th of each month for the following month, so for March visits, book on February 10th at 10 AM Japan time.
user: Actually, Sarah already booked Ghibli Museum tickets for March 18th at 2 PM. She set an alarm for February 10th at 8 PM EST (which is 10 AM JST the next day). We'll take the Chuo Line from Shinjuku to Mitaka Station, then walk 15 minutes to the museum.
assistant: Perfect! That works well since you're still in Tokyo on the 18th. The Chuo Line is convenient. What about transportation passes?
user: We're getting the 7-day Japan Rail Pass starting March 16th, which covers us through March 22nd. It costs $350 each. For local Tokyo transport, we'll use Suica cards that we'll buy at the airport. Sarah said we should load at least 5000 yen initially.
assistant: Smart! The JR Pass will cover your Shinkansen trips and most JR trains. Suica cards are convenient for subways and buses. 5000 yen should last a few days depending on usage.
user: One important thing - we're both strict vegetarians, no fish, no meat, no dashi. Sarah is also allergic to shellfish. Will this be difficult?
assistant: It will be challenging but doable with planning. Many Japanese dishes use dashi (fish-based stock), so you'll need to be very careful. In Tokyo, I recommend Ain Soph Ripple in Ginza for vegan options, or T's Tantan in Tokyo Station. In Kyoto, look for shojin ryori (Buddhist vegetarian cuisine) at temples like Tenryu-ji. Always ask about ingredients.
user: That's really helpful! We'll make a list of those restaurants. Oh, I almost forgot - we're meeting my friend Mike in Osaka on March 22nd at 3 PM. He's flying in from Seoul on Korean Air Flight KE621, arriving at 1:45 PM at Kansai Airport.
assistant: That's nice! Mike will join you for part of the trip. Kansai Airport is about 50 minutes from central Osaka by train. Where are you planning to meet?
user: We're meeting at Osaka Station, specifically at the HEP Five Ferris Wheel entrance on the 7th floor. Mike suggested it because it's easy to find. He's staying at the Hotel Nikko Osaka for 2 nights, March 22nd and 23rd.
assistant: HEP Five is a great landmark! Osaka Station can be busy, so having a specific meeting point is smart. Hotel Nikko is well-located. What are your plans in Osaka?
user: On March 22nd evening, we're going to Dotonbori for dinner and exploring. Mike knows a great okonomiyaki place called Mizuno. On March 23rd, Mike suggested Universal Studios Japan since he has annual passes. Sarah and I need to buy tickets - we're thinking of getting the Express Pass 4 for about $150 each to skip lines.
assistant: USJ is fantastic! The Express Pass is worth it, especially for popular attractions like The Wizarding World of Harry Potter and the new Super Nintendo World. Book tickets online in advance. Mizuno is famous for okonomiyaki - though you'll need to check if they can accommodate vegetarian diets.
user: Good point about Mizuno. We'll call ahead. After USJ on March 23rd, we're all going to a baseball game - the Hanshin Tigers vs Tokyo Giants at Koshien Stadium at 6 PM. Mike got us tickets in section 12, row 25.
assistant: Japanese baseball games are a unique experience! Koshien Stadium is historic. The atmosphere is amazing. Are you planning to stay in Osaka that night?
user: Yes, we're staying at a budget hotel called Hotel Dotonbori near the canal for March 22nd and 23rd. It's about $80 per night. Then on March 24th, we're taking the Shinkansen back to Tokyo for our flight home on March 25th.
assistant: Hotel Dotonbori is convenient for nightlife. What time is your return flight?
user: Our return flight is JAL Flight 005 on March 25th, departing Tokyo Narita at 11:30 AM. We need to be at the airport by 9:30 AM, so we'll take the Narita Express from Tokyo Station at 7:45 AM on March 25th.
assistant: That's an early start! The Narita Express takes about an hour. Make sure to account for traffic. Are you staying in Tokyo on the 24th?
user: Yes, we're staying one more night at a different hotel - the Tokyo Station Hotel, just for convenience since we're leaving early. We'll check in on March 24th and check out on March 25th morning.
assistant: Tokyo Station Hotel is perfect for early departures - it's connected to the station. One last thing - have you thought about travel insurance and what to do if you get lost?
user: Yes, we both have travel insurance through Allianz. Sarah speaks basic Japanese - she took 2 years of classes. I don't speak any, but I downloaded Google Translate offline. We also have portable WiFi devices rented from Japan Wireless for $8 per day each.
assistant: That's excellent preparation! Portable WiFi is essential. Google Translate works well, especially the camera feature for translating signs. Sarah's Japanese will definitely help. Is there anything else you're concerned about?
user: Actually, yes. I'm worried about money. We're bringing some cash, but mostly using credit cards. Sarah has a Chase Sapphire card, and I have an American Express Platinum. Do most places accept cards?
assistant: Major cities like Tokyo, Kyoto, and Osaka accept cards at most hotels, restaurants, and shops. However, many smaller restaurants, temples, and local shops are cash-only. I'd recommend carrying at least 20,000-30,000 yen per person. ATMs at 7-Eleven convenience stores accept foreign cards and are available 24/7.
user: Perfect! We'll make sure to have enough cash. I think we have everything planned now. Thanks so much for all your detailed help!
assistant: You're very welcome! You've planned an amazing trip. Don't forget to try matcha ice cream, visit a traditional onsen if you get a chance, and enjoy the cherry blossoms if they're blooming. Have a wonderful time in Japan!""",
    "metadata": {
        "timestamp": "2024-01-15T14:30:00",
        "topic": "Japan Travel Planning",
        "language": "en-US",
    }
}


def process_session(
    memory: MemorySystem,
    adapter: MemoryAdapter,
    evaluator: MemoryEvaluator,
    challenger: MemoryChallenger,
    user_id: str,
    session_id: str,
    session_dialogue: str,
    output_dir: str = "output",
) -> Dict[str, Any]:
    """Process a single session: build memory -> evaluate -> reconstruct if needed -> save."""
    result: Dict[str, Any] = {
        "session_id": session_id,
        "user_id": user_id,
        "success": False,
        "pass_rate": 0.0,
        "reconstructed": False,
        "memory_file": None,
    }

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    try:
        print(f"\n[{user_id}] Processing session {session_id}...")
        print(f"[{user_id}] Building memory from dialogue...")
        
        build_result = memory.build_memory(user_id, session_dialogue)
        if build_result.get("status") == "built":
            print(f"[{user_id}] Memory built successfully")
            print(f"[{user_id}] Summary: {build_result.get('summary', '')[:100]}...")
        else:
            print(f"[{user_id}] Memory building failed: {build_result.get('error', 'unknown error')}")

        print(f"[{user_id}] Evaluating memory quality...")

        eval_result, need_reconstruct = evaluator.evaluate_session_memories(memory, user_id, session_dialogue)

        result["pass_rate"] = eval_result["summary"]["pass_rate"]
        result["qa_count"] = eval_result["summary"]["qa_pairs_count"]
        result["passed"] = eval_result["summary"]["passed"]
        result["failed"] = eval_result["summary"]["failed"]

        print(f"\n[{user_id}] Initial Evaluation Summary:")
        print(f"  Total Questions: {result['qa_count']}")
        print(f"  Passed: {result['passed']}")
        print(f"  Failed: {result['failed']}")
        print(f"  Pass Rate: {result['pass_rate']:.2f}%")
        
        if eval_result.get("failed_qa"):
            print(f"\n[{user_id}] Failed Questions:")
            for idx, failed_item in enumerate(eval_result["failed_qa"], 1):
                print(f"  {idx}. Q: {failed_item['question']}")
                print(f"     Expected: {failed_item['true_answer']}")
                print(f"     Got: {failed_item['retrieved_answer']}")
                print(f"     Similarity: {failed_item['similarity']:.4f}")

        initial_pass_rate = result["pass_rate"]
        result["initial_pass_rate"] = initial_pass_rate

        if need_reconstruct:
            print(f"[{user_id}] Pass rate {result['pass_rate']:.2f}% < {PASS_RATE_THRESHOLD}%, reconstructing...")
            failed_qa = eval_result.get("failed_qa", [])
            failed_reasons = eval_result.get("error_analysis", {}).get("error_types", [])

            filtered_dialogue = adapter.update(memory, user_id, session_dialogue, failed_qa, failed_reasons)
            original_lines = len(session_dialogue.split("\n"))
            filtered_lines = len(filtered_dialogue.split("\n"))
            print(f"[{user_id}] Filtered dialogue: {original_lines} -> {filtered_lines} lines")

            reconstruct_success = adapter.reconstruct(memory, user_id, filtered_dialogue, failed_qa)
            result["reconstructed"] = reconstruct_success

            if reconstruct_success:
                memory_file = output_path / f"{user_id}_{session_id}_reconstructed_memory.json"
                if isinstance(memory, SimpleMemorySystem):
                    memory.save_to_file(user_id, str(memory_file))
                result["memory_file"] = str(memory_file)
                print(f"[{user_id}] Reconstructed memory saved to {memory_file}")
                
                print(f"\n[{user_id}] Re-evaluating memory after reconstruction...")
                eval_result_after, need_reconstruct_after = evaluator.evaluate_session_memories(
                    memory, user_id, session_dialogue
                )
                
                result["after_reconstruct"] = {
                    "pass_rate": eval_result_after["summary"]["pass_rate"],
                    "qa_count": eval_result_after["summary"]["qa_pairs_count"],
                    "passed": eval_result_after["summary"]["passed"],
                    "failed": eval_result_after["summary"]["failed"],
                }
                
                print(f"\n[{user_id}] Post-Reconstruction Evaluation Summary:")
                print(f"  Total Questions: {result['after_reconstruct']['qa_count']}")
                print(f"  Passed: {result['after_reconstruct']['passed']}")
                print(f"  Failed: {result['after_reconstruct']['failed']}")
                print(f"  Pass Rate: {result['after_reconstruct']['pass_rate']:.2f}%")
                
                improvement = result["after_reconstruct"]["pass_rate"] - initial_pass_rate
                print(f"\n[{user_id}] Improvement: {improvement:+.2f}% (from {initial_pass_rate:.2f}% to {result['after_reconstruct']['pass_rate']:.2f}%)")
                
                if eval_result_after.get("failed_qa"):
                    print(f"\n[{user_id}] Remaining Failed Questions After Reconstruction:")
                    for idx, failed_item in enumerate(eval_result_after["failed_qa"], 1):
                        print(f"  {idx}. Q: {failed_item['question']}")
                        print(f"     Expected: {failed_item['true_answer']}")
                        print(f"     Got: {failed_item['retrieved_answer']}")
                        print(f"     Similarity: {failed_item['similarity']:.4f}")
        else:
            print(f"[{user_id}] Pass rate {result['pass_rate']:.2f}% >= {PASS_RATE_THRESHOLD}%, saving original memory...")
            memory_file = output_path / f"{user_id}_{session_id}_original_memory.json"
            if isinstance(memory, SimpleMemorySystem):
                memory.save_to_file(user_id, str(memory_file))
            result["memory_file"] = str(memory_file)
            print(f"[{user_id}] Original memory saved to {memory_file}")

        result["success"] = True
        print(f"[{user_id}] Session {session_id} completed successfully")

    except Exception as e:
        result["error"] = str(e)
        print(f"[{user_id}] Session {session_id} failed: {e}")
        import traceback
        traceback.print_exc()

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simple session memory evaluation example")
    parser.add_argument("--dialogue", help="Dialogue JSON file or inline JSON")
    parser.add_argument("--user-id", default="test_user", help="User ID")
    parser.add_argument("--session-id", default="session_1", help="Session ID")
    parser.add_argument("--output-dir", default="output", help="Output directory for memory files")
    args = parser.parse_args()

    if args.dialogue:
        if Path(args.dialogue).exists():
            dialogue_text = Path(args.dialogue).read_text(encoding="utf-8")
            try:
                dialogue_data = json.loads(dialogue_text)
                if isinstance(dialogue_data, list):
                    session_dialogue = "\n".join([
                        f"{item.get('role', item.get('speaker', 'user'))}: {item.get('content', item.get('text', ''))}"
                        for item in dialogue_data
                    ])
                elif isinstance(dialogue_data, dict) and "session_dialogue" in dialogue_data:
                    session_dialogue = dialogue_data["session_dialogue"]
                    if not args.user_id or args.user_id == "test_user":
                        args.user_id = dialogue_data.get("user_id", args.user_id)
                    if not args.session_id or args.session_id == "session_1":
                        args.session_id = dialogue_data.get("session_id", args.session_id)
                else:
                    session_dialogue = dialogue_text
            except json.JSONDecodeError:
                session_dialogue = dialogue_text
        else:
            session_dialogue = args.dialogue
    else:
        # 使用示例数据
        session_dialogue = EXAMPLE_SESSION_DATA["session_dialogue"]
        if not args.user_id or args.user_id == "test_user":
            args.user_id = EXAMPLE_SESSION_DATA["user_id"]
        if not args.session_id or args.session_id == "session_1":
            args.session_id = EXAMPLE_SESSION_DATA["session_id"]

    memory = SimpleMemorySystem(storage_dir="memory_storage")
    adapter = SimpleMemoryAdapter()
    challenger = MemoryChallenger()
    evaluator = MemoryEvaluator(challenger=challenger)

    result = process_session(
        memory=memory,
        adapter=adapter,
        evaluator=evaluator,
        challenger=challenger,
        user_id=args.user_id,
        session_id=args.session_id,
        session_dialogue=session_dialogue,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("Session Processing Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
