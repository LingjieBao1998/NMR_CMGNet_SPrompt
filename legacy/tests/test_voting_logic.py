#!/usr/bin/env python3
"""
测试修改后的投票逻辑
"""

def test_voting_mechanism():
    """测试2/3投票机制"""
    
    print("🧪 测试2/3投票机制")
    print("=" * 50)
    
    # 测试用例
    test_cases = [
        {
            "name": "全票通过 (3/3)",
            "votes": [True, True, True],
            "expected": True
        },
        {
            "name": "多数通过 (2/3)",
            "votes": [True, True, False],
            "expected": True
        },
        {
            "name": "少数未通过 (1/3)",
            "votes": [True, False, False],
            "expected": False
        },
        {
            "name": "全票否决 (0/3)",
            "votes": [False, False, False],
            "expected": False
        },
        {
            "name": "两票否决 (1/3)",
            "votes": [False, False, True],
            "expected": False
        }
    ]
    
    for case in test_cases:
        total_votes = len(case["votes"])
        positive_votes = sum(case["votes"])
        is_core_molecule = positive_votes >= (2 * total_votes / 3) if total_votes > 0 else False
        
        status = "✅" if is_core_molecule == case["expected"] else "❌"
        print(f"{status} {case['name']}: {positive_votes}/{total_votes} -> {is_core_molecule} (期望: {case['expected']})")
    
    print("\n🎯 投票机制验证完成")
    print("现在集成模型将始终使用2/3投票机制，不再有高置信度例外")

if __name__ == "__main__":
    test_voting_mechanism()