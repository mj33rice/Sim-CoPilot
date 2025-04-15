import unittest
import re
from helper_functions.post_gen_process import is_code_line

class TestPythonCodeLine(unittest.TestCase):
    def test_control_structures(self):
        # Test various Python control structures
        self.assertTrue(is_code_line("if x > 5:", "Python")[0])
        self.assertTrue(is_code_line("else:", "Python")[0])
        self.assertTrue(is_code_line("elif x < 10:", "Python")[0])
        self.assertTrue(is_code_line("for i in range(10):", "Python")[0])
        self.assertTrue(is_code_line("while True:", "Python")[0])
        self.assertTrue(is_code_line("try:", "Python")[0])
        self.assertTrue(is_code_line("except ValueError:", "Python")[0])
        self.assertTrue(is_code_line("except:", "Python")[0])
        self.assertTrue(is_code_line("finally:", "Python")[0])
        self.assertTrue(is_code_line("with open('file.txt') as f:", "Python")[0])
        self.assertTrue(is_code_line("def function(args):", "Python")[0])
        self.assertTrue(is_code_line("class MyClass:", "Python")[0])
        
    def test_statements(self):
        # Test various Python statements
        self.assertTrue(is_code_line("return result", "Python")[0])
        self.assertTrue(is_code_line("return", "Python")[0])
        self.assertTrue(is_code_line("break", "Python")[0]) 
        self.assertTrue(is_code_line("continue", "Python")[0])
        self.assertTrue(is_code_line("pass", "Python")[0])
        self.assertTrue(is_code_line("raise ValueError('Invalid input')", "Python")[0])
        self.assertTrue(is_code_line("assert x > 0, 'x must be positive'", "Python")[0])
        
    def test_assignments(self):
        # Test various assignment operations
        self.assertTrue(is_code_line("x = 5", "Python")[0])
        self.assertTrue(is_code_line("x += 1", "Python")[0])
        self.assertTrue(is_code_line("x -= 2", "Python")[0])
        self.assertTrue(is_code_line("x *= 3", "Python")[0])
        self.assertTrue(is_code_line("x /= 4", "Python")[0])
        self.assertTrue(is_code_line("x %= 5", "Python")[0])
        self.assertTrue(is_code_line("x **= 2", "Python")[0])
        self.assertTrue(is_code_line("x //= 3", "Python")[0])
        self.assertTrue(is_code_line("x, y = 1, 2", "Python")[0])
        self.assertTrue(is_code_line("batch_size = tf.shape(states)[0]", "Python")[0])
        
    def test_function_calls(self):
        # Test function calls and indexing
        self.assertTrue(is_code_line("print('Hello')", "Python")[0])
        self.assertTrue(is_code_line("calculate(x, y, z)", "Python")[0])
        self.assertTrue(is_code_line("result = func(a, b=3)", "Python")[0])
        self.assertTrue(is_code_line("data[0]", "Python")[0])
        self.assertTrue(is_code_line("data['key']", "Python")[0])
        self.assertTrue(is_code_line("obj.method()", "Python")[0])
        self.assertTrue(is_code_line("skill_seq = tf.argmax(skill_seq, axis=-1)", "Python")[0])
        
    def test_data_structures(self):
        # Test lines with data structure operations
        self.assertTrue(is_code_line("[1, 2, 3]", "Python")[0])
        self.assertTrue(is_code_line("{'a': 1, 'b': 2}", "Python")[0])
        self.assertTrue(is_code_line("(x, y, z)", "Python")[0])
        self.assertTrue(is_code_line("[]", "Python")[0])
        self.assertTrue(is_code_line("{}", "Python")[0])
        self.assertTrue(is_code_line("()", "Python")[0])
        self.assertTrue(is_code_line("arr = [1, 2,", "Python")[0])  # Line ending with comma
        
    def test_expressions(self):
        # Test lines with expressions
        self.assertTrue(is_code_line("x + y", "Python")[0])
        self.assertTrue(is_code_line("x - y", "Python")[0])
        self.assertTrue(is_code_line("x * y", "Python")[0])
        self.assertTrue(is_code_line("x / y", "Python")[0])
        self.assertTrue(is_code_line("x % y", "Python")[0])
        self.assertTrue(is_code_line("x == y", "Python")[0])
        self.assertTrue(is_code_line("x != y", "Python")[0])
        self.assertTrue(is_code_line("x <= y", "Python")[0])
        self.assertTrue(is_code_line("x >= y", "Python")[0])
        self.assertTrue(is_code_line("x < y", "Python")[0])
        self.assertTrue(is_code_line("x > y", "Python")[0])
        
    def test_indentation(self):
        # Test indented code
        self.assertTrue(is_code_line("    x = 5", "Python")[0])
        self.assertTrue(is_code_line("        if condition:", "Python")[0])
        self.assertTrue(is_code_line("            return result", "Python")[0])
        self.assertTrue(is_code_line("\tx = 5", "Python")[0])  # Tab indentation
        
    def test_comments(self):
        # Test comments (should not be identified as code if store_comment=False)
        self.assertFalse(is_code_line("# This is a comment", "Python", store_comment=False)[0])
        self.assertFalse(is_code_line("    # Indented comment", "Python", store_comment=False)[0])
        
        # Test block comments
        result, is_block, next_block = is_code_line('"""', "Python", store_comment=False)
        self.assertTrue(is_block)
        self.assertTrue(next_block)
        
        result, is_block, next_block = is_code_line('"""Docstring"""', "Python", store_comment=False) 
        self.assertTrue(is_block)
        self.assertFalse(next_block)
        
    def test_mixed_code_and_comments(self):
        # Test mixed code and comments
        self.assertTrue(is_code_line("x = 5  # Assignment", "Python")[0])
        self.assertTrue(is_code_line("if True:  # Always executes", "Python")[0])
        
    def test_edge_cases(self):
        # Test edge cases
        self.assertFalse(is_code_line("", "Python")[0])  # Empty line
        self.assertFalse(is_code_line("   ", "Python")[0])  # Whitespace-only line
        self.assertTrue(is_code_line("lambda x: x*2", "Python")[0])  # Lambda expressions
        self.assertTrue(is_code_line("@decorator", "Python")[0])  # Decorators
        
    def test_special_cases(self):
        # Test the 'else:' construct (which was reported as problematic)
        self.assertTrue(is_code_line("else:", "Python")[0])
        
        # Very important case from example
        self.assertTrue(is_code_line("batch_size = tf.shape(states)[0]", "Python")[0])
        self.assertTrue(is_code_line("states_goals = tf.reshape(states[:, :3 * num_objs], (batch_size, num_objs, 3))", "Python")[0])
        self.assertTrue(is_code_line("goals = tf.gather(states_goals, obj_identifiers, batch_dims=1)", "Python")[0])
    
    def test_complex_llm_generated_code(self):
        """Test complex or incomplete code patterns often generated by LLMs"""
        # Incomplete function definitions
        self.assertTrue(is_code_line("def process_data(data", "Python")[0])
        self.assertTrue(is_code_line("def validate_input(x: int,", "Python")[0])
        
        # Complex list/dict comprehensions
        self.assertTrue(is_code_line("[x**2 for x in range(10) if x % 2 == 0", "Python")[0])
        self.assertTrue(is_code_line("{k: v for k, v in zip(keys, values) if k not", "Python")[0])
        
        # Modern Python features
        self.assertTrue(is_code_line("result: Dict[str, List[int]] = {}", "Python")[0])  # Type hints
        self.assertTrue(is_code_line("if (count := len(items)) > 10:", "Python")[0])  # Walrus operator
        self.assertTrue(is_code_line("async def fetch_data(url):", "Python")[0])  # Async function
        self.assertTrue(is_code_line("await response.json()", "Python")[0])  # Await expression
        
        # Complex string operations
        self.assertTrue(is_code_line("f\"User {user.name} has {user.points:,} points\"", "Python")[0])  # f-strings
        self.assertTrue(is_code_line("text = \"\"\"Multiple", "Python")[0])  # Incomplete multi-line string
        
        # Complex slicing/indexing
        self.assertTrue(is_code_line("matrix[2:5, 3:7]", "Python")[0])  # NumPy-style slicing
        self.assertTrue(is_code_line("data.loc[:, ['A', 'B']]", "Python")[0])  # Pandas-style indexing
        
        # Nested data structures
        self.assertTrue(is_code_line("config = {'params': {'learning_rate': 0.01, 'layers': [64, 32", "Python")[0])
        self.assertTrue(is_code_line("result = [[func(i, j) for j in range(cols)]", "Python")[0])
        
        # Generator expressions 
        self.assertTrue(is_code_line("(x for x in range(100) if is_prime(x))", "Python")[0])
        
        # Complex decorator patterns
        self.assertTrue(is_code_line("@app.route('/api/v1/users/<int:user_id>', methods=['GET', 'POST'])", "Python")[0])
        
        # Context managers with unusual structure
        self.assertTrue(is_code_line("with open(file_path, 'r') as f, ", "Python")[0])
        
        # Complex boolean expressions
        self.assertTrue(is_code_line("if all(x > 0 for x in values) and not any(y < threshold for y in limits):", "Python")[0])
        
        # ML-specific code patterns (TensorFlow, PyTorch)
        self.assertTrue(is_code_line("model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))", "Python")[0])
        self.assertTrue(is_code_line("outputs = torch.nn.functional.softmax(logits, dim=1)", "Python")[0])
    
    def test_non_code_text_mentioning_code(self):
        """Test natural language text that mentions code but isn't code itself"""
        # Text describing code
        self.assertFalse(is_code_line("Let me explain how the function calculate() works in this algorithm.", "Python", store_comment=False)[0])
        self.assertFalse(is_code_line("You need to update the if statement to include the new condition.", "Python", store_comment=False)[0])
        self.assertFalse(is_code_line("The code should return a value when x > 10.", "Python", store_comment=False)[0])
        self.assertFalse(is_code_line("Define a new class called DataProcessor to handle the input.", "Python", store_comment=False)[0])
        
        # Text with code-like fragments
        self.assertFalse(is_code_line("In Python, you'd write something like if x > 5: to check conditions.", "Python", store_comment=False)[0])
        self.assertFalse(is_code_line("I think the function my_func(x, y) needs to handle negative inputs better.", "Python", store_comment=False)[0])
        self.assertFalse(is_code_line("Try adding a break statement to exit the loop early when found.", "Python", store_comment=False)[0])
        self.assertFalse(is_code_line("The variable x should be set to 5, not 10, to fix the issue.", "Python", store_comment=False)[0])
        
        # Code instructions disguised as text
        self.assertFalse(is_code_line("You should use a dictionary like {'key': value} for this task.", "Python", store_comment=False)[0])
        self.assertFalse(is_code_line("Add 'return result' at the end of the function.", "Python", store_comment=False)[0])
        self.assertFalse(is_code_line("Replace 'if data:' with 'if data is not None:' for clarity.", "Python", store_comment=False)[0])
        
        # Complex descriptions with multiple code references
        self.assertFalse(is_code_line("First create a list comprehension [x for x in range(10)], then filter it with an if condition.", "Python", store_comment=False)[0])
        self.assertFalse(is_code_line("The class should inherit from BaseModel and override the process() method.", "Python", store_comment=False)[0])
        self.assertFalse(is_code_line("Try using try/except blocks to catch ValueError and TypeError separately.", "Python", store_comment=False)[0])

class TestJavaCodeLine(unittest.TestCase):
    def test_control_structures(self):
        # Test various Java control structures
        self.assertTrue(is_code_line("if (x > 5) {", "Java")[0])
        self.assertTrue(is_code_line("else {", "Java")[0])
        self.assertTrue(is_code_line("else if (x < 10) {", "Java")[0])
        self.assertTrue(is_code_line("for (int i = 0; i < 10; i++) {", "Java")[0])
        self.assertTrue(is_code_line("while (true) {", "Java")[0])
        self.assertTrue(is_code_line("do {", "Java")[0])
        self.assertTrue(is_code_line("} while (condition);", "Java")[0])
        self.assertTrue(is_code_line("switch (value) {", "Java")[0])
        self.assertTrue(is_code_line("case 1:", "Java")[0])
        self.assertTrue(is_code_line("default:", "Java")[0])
        self.assertTrue(is_code_line("try {", "Java")[0])
        self.assertTrue(is_code_line("} catch (Exception e) {", "Java")[0])
        self.assertTrue(is_code_line("} finally {", "Java")[0])
        
    def test_statements(self):
        # Test various Java statements
        self.assertTrue(is_code_line("return result;", "Java")[0])
        self.assertTrue(is_code_line("return;", "Java")[0])
        self.assertTrue(is_code_line("break;", "Java")[0]) 
        self.assertTrue(is_code_line("continue;", "Java")[0])
        self.assertTrue(is_code_line("throw new IllegalArgumentException(\"Invalid input\");", "Java")[0])
        self.assertTrue(is_code_line("assert x > 0 : \"x must be positive\";", "Java")[0])
        self.assertTrue(is_code_line("System.out.println(\"Hello World\");", "Java")[0])
        
    def test_declarations(self):
        # Test class, interface, method declarations
        self.assertTrue(is_code_line("public class MyClass {", "Java")[0])
        self.assertTrue(is_code_line("private class InnerClass {", "Java")[0])
        self.assertTrue(is_code_line("public interface MyInterface {", "Java")[0])
        self.assertTrue(is_code_line("public static void main(String[] args) {", "Java")[0])
        self.assertTrue(is_code_line("private int calculateSum(int a, int b) {", "Java")[0])
        self.assertTrue(is_code_line("protected String getName() {", "Java")[0])
        self.assertTrue(is_code_line("public MyClass() {", "Java")[0])  # Constructor
        self.assertTrue(is_code_line("@Override", "Java")[0])  # Annotation
        self.assertTrue(is_code_line("enum Status { ACTIVE, INACTIVE, PENDING }", "Java")[0])
        
    def test_variable_declarations(self):
        # Test variable declarations and assignments
        self.assertTrue(is_code_line("int x = 5;", "Java")[0])
        self.assertTrue(is_code_line("final double PI = 3.14159;", "Java")[0])
        self.assertTrue(is_code_line("String name = \"John\";", "Java")[0])
        self.assertTrue(is_code_line("List<String> names = new ArrayList<>();", "Java")[0])
        self.assertTrue(is_code_line("Map<String, Integer> scores = new HashMap<>();", "Java")[0])
        self.assertTrue(is_code_line("int[] numbers = {1, 2, 3, 4, 5};", "Java")[0])
        self.assertTrue(is_code_line("Student student = new Student(\"Alice\", 20);", "Java")[0])
        
    def test_assignments_and_operations(self):
        # Test assignments and operations
        self.assertTrue(is_code_line("x = 10;", "Java")[0])
        self.assertTrue(is_code_line("count += 1;", "Java")[0])
        self.assertTrue(is_code_line("total -= amount;", "Java")[0])
        self.assertTrue(is_code_line("value *= factor;", "Java")[0])
        self.assertTrue(is_code_line("result /= divisor;", "Java")[0])
        self.assertTrue(is_code_line("count++;", "Java")[0])
        self.assertTrue(is_code_line("index--;", "Java")[0])
        self.assertTrue(is_code_line("isValid = (count > 0 && name != null);", "Java")[0])
        
    def test_method_calls(self):
        # Test method calls
        self.assertTrue(is_code_line("calculateTotal();", "Java")[0])
        self.assertTrue(is_code_line("int sum = add(5, 3);", "Java")[0])
        self.assertTrue(is_code_line("System.out.println(\"Hello World\");", "Java")[0])
        self.assertTrue(is_code_line("String name = user.getName();", "Java")[0])
        self.assertTrue(is_code_line("data.add(newItem);", "Java")[0])
        self.assertTrue(is_code_line("return Math.max(a, b);", "Java")[0])
        self.assertTrue(is_code_line("String input = scanner.nextLine();", "Java")[0])
        
    def test_control_flow(self):
        # Test more complex control flow
        self.assertTrue(is_code_line("if (age >= 18) {", "Java")[0])
        self.assertTrue(is_code_line("} else if (age >= 13) {", "Java")[0])
        self.assertTrue(is_code_line("} else {", "Java")[0])
        self.assertTrue(is_code_line("for (String item : items) {", "Java")[0])  # Enhanced for loop
        self.assertTrue(is_code_line("while ((line = reader.readLine()) != null) {", "Java")[0])
        
    def test_imports_and_packages(self):
        # Test imports and package declarations
        self.assertTrue(is_code_line("import java.util.List;", "Java")[0])
        self.assertTrue(is_code_line("import java.util.*;", "Java")[0])
        self.assertTrue(is_code_line("import static java.lang.Math.PI;", "Java")[0])
        self.assertTrue(is_code_line("package com.example.myapp;", "Java")[0])
        
    def test_advanced_features(self):
        # Test more advanced Java features
        self.assertTrue(is_code_line("Runnable r = () -> System.out.println(\"Lambda\");", "Java")[0])  # Lambda
        self.assertTrue(is_code_line("Stream<Integer> stream = numbers.stream().filter(n -> n > 0);", "Java")[0])  # Stream API
        self.assertTrue(is_code_line("var result = service.processData();", "Java")[0])  # Type inference (Java 10+)
        self.assertTrue(is_code_line("String result = switch (status) {", "Java")[0])  # Switch expression (Java 14+)
        self.assertTrue(is_code_line("record Person(String name, int age) {}", "Java")[0])  # Records (Java 16+)
        
    def test_comments(self):
        # Test comments
        self.assertFalse(is_code_line("// This is a comment", "Java", store_comment=False)[0])
        self.assertFalse(is_code_line("    // Indented comment", "Java", store_comment=False)[0])
        
        # Test block comments
        result, is_block, next_block = is_code_line("/*", "Java", store_comment=False)
        self.assertTrue(is_block)
        self.assertTrue(next_block)
        
        result, is_block, next_block = is_code_line("/* Multi-line comment */", "Java", store_comment=False)
        self.assertTrue(is_block)
        self.assertFalse(next_block)
        
        self.assertTrue(is_code_line("int count = 0; // Initialize counter", "Java")[0])  # Code with end-line comment
        
    def test_edge_cases(self):
        # Test edge cases
        self.assertFalse(is_code_line("", "Java")[0])  # Empty line
        self.assertFalse(is_code_line("   ", "Java")[0])  # Whitespace-only line
        self.assertTrue(is_code_line("}", "Java")[0])  # Just a closing brace
        self.assertTrue(is_code_line("{", "Java")[0])  # Just an opening brace
        self.assertTrue(is_code_line(");", "Java")[0])  # Just a closing parenthesis with semicolon
        
    def test_incomplete_code(self):
        # Test incomplete code (common in LLM outputs)
        self.assertTrue(is_code_line("public void processData(String data", "Java")[0])  # Incomplete method signature
        self.assertTrue(is_code_line("if (isValid && ", "Java")[0])  # Incomplete condition
        self.assertTrue(is_code_line("List<Map<String, Object>> ", "Java")[0])  # Just a complex type
        self.assertTrue(is_code_line("return service.findById(", "Java")[0])  # Incomplete method call
        self.assertTrue(is_code_line("for (User user : userList) ", "Java")[0])  # Incomplete loop
        
    def test_complex_code_patterns(self):
        # Test more complex Java code patterns
        self.assertTrue(is_code_line("Map<String, List<Integer>> groupedData = new HashMap<>();", "Java")[0])
        self.assertTrue(is_code_line("results.stream().filter(r -> r.isValid()).map(Result::getValue).collect(Collectors.toList());", "Java")[0])
        self.assertTrue(is_code_line("public <T extends Comparable<T>> T findMax(List<T> list) {", "Java")[0])  # Generic method
        self.assertTrue(is_code_line("new Thread(() -> { processing(); }).start();", "Java")[0])  # Anonymous class with lambda
        
    def test_non_code_text_mentioning_code(self):
        """Test natural language text that mentions Java code but isn't code itself"""
        # Text describing code
        self.assertFalse(is_code_line("Let me explain how the Java method works in this class.", "Java", store_comment=False)[0])
        self.assertFalse(is_code_line("You need to update the if statement to include the null check.", "Java", store_comment=False)[0])
        self.assertFalse(is_code_line("The code should throw an exception when input is invalid.", "Java", store_comment=False)[0])
        self.assertFalse(is_code_line("Create a new class that implements the Runnable interface.", "Java", store_comment=False)[0])
        
        # Text with code-like fragments
        self.assertFalse(is_code_line("In Java, you'd write something like if (x > 5) { to check conditions.", "Java", store_comment=False)[0])
        self.assertFalse(is_code_line("I think the method processData(String input) needs better error handling.", "Java", store_comment=False)[0])
        self.assertFalse(is_code_line("Add a break statement to exit the loop early when the value is found.", "Java", store_comment=False)[0])
        self.assertFalse(is_code_line("The variable count should be set to 0, not null, to fix the issue.", "Java", store_comment=False)[0])
    
    def test_object_property_assignments(self):
        # Test object property assignments
        self.assertTrue(is_code_line("newOne.distance = distance;", "Java")[0])
        self.assertTrue(is_code_line("user.name = userName;", "Java")[0])
        self.assertTrue(is_code_line("this.count = 0;", "Java")[0])
        self.assertTrue(is_code_line("node.next = null;", "Java")[0])
        self.assertTrue(is_code_line("response.data.items = [];", "Java")[0])
        
    def test_incomplete_expressions(self):
        # Test incomplete expressions that should still be recognized as code
        self.assertTrue(is_code_line("large", "Java")[0])  # Single variable
        self.assertTrue(is_code_line("counter", "Java")[0])  # Single variable
        self.assertTrue(is_code_line("myObj.property", "Java")[0])  # Property access
        self.assertTrue(is_code_line("array[index]", "Java")[0])  # Array access
        self.assertTrue(is_code_line("obj.getValues()", "Java")[0])  # Method call without semicolon

if __name__ == '__main__':
    unittest.main()
