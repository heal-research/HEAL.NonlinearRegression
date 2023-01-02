using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Text;

namespace HEAL.Expressions.Parser {
  public enum TokenEnum { None, Plus, Minus, Mul, Div, Pow, LeftPar, RightPar, Comma, Parameter, Constant, Ident, Eof };
  public class ExprParser {

    private readonly Scanner scanner;
    private TokenEnum Sy => scanner.Token;
    private void NextSy() { scanner.NextToken(); }

    private List<double> parameterValues = new List<double>();
    public double[] ParameterValues => parameterValues.ToArray();

    private readonly string[] variableNames;
    private readonly ParameterExpression variableSymbol;
    private readonly ParameterExpression parameterSymbol;

    // the parser returns a lambda expression (variableSym, parameterSym) => f(variableSym, parameterSym)
    // variableSym and parameterSym must have double[] type
    public ExprParser(string input, string[] variableNames, ParameterExpression variableSymbol, ParameterExpression parameterSymbol) {
      scanner = new Scanner(input);
      scanner.NextToken();

      this.variableNames = variableNames;
      this.variableSymbol = variableSymbol;
      this.parameterSymbol = parameterSymbol;
    }

    public Expression<Expr.ParametricFunction> Parse() {
      return Expression.Lambda<Expr.ParametricFunction>(Expr(), parameterSymbol, variableSymbol);
    }


    // LL(1) grammar:
    // G(Expr):
    // Expr      = Term { ('+' | '-') Term
    // Term      = Fact { ('*' | '/') Fact) }
    // Fact      = ['+' | '-' ]
    //             (ident | constant | parameter
    //              | '(' Expr ')'
    //              | ident ParamList                           // function call
    //             ) [ ('**' | '^') Fact ]
    // ParamList = '(' Expr { ',' Expr } ')' 


    // A simple recursive descent parser
    private Expression Expr() {
      var expr = Term();
      while (Sy == TokenEnum.Minus || Sy == TokenEnum.Plus) {
        if (Sy == TokenEnum.Minus) {
          NextSy();
          expr = Expression.MakeBinary(ExpressionType.Subtract, expr, Term());
        } else if (Sy == TokenEnum.Plus) {
          NextSy();
          expr = Expression.MakeBinary(ExpressionType.Add, expr, Term());
        }
      }
      return expr;
    }

    private Expression Term() {
      var expr = Fact();
      while (Sy == TokenEnum.Mul || Sy == TokenEnum.Div) {
        if (Sy == TokenEnum.Mul) {
          NextSy();
          expr = Expression.MakeBinary(ExpressionType.Multiply, expr, Fact());
        } else if (Sy == TokenEnum.Div) {
          NextSy();
          expr = Expression.MakeBinary(ExpressionType.Divide, expr, Fact());
        }
      }
      return expr;
    }

    private Expression Fact() {
      Expression expr;
      var negate = false;
      if (Sy == TokenEnum.Minus || Sy == TokenEnum.Plus) {
        negate = Sy == TokenEnum.Minus;
        NextSy();
      }

      switch (Sy) {
        case TokenEnum.Ident: {
            var idStr = scanner.IdStr;
            NextSy();
            if (Sy == TokenEnum.LeftPar) {
              // identifier ParamList (= funcCall)
              var parameters = ParamList();
              expr = GetMethodCall(idStr, parameters);
            } else {
              // identifier
              var idx = Array.IndexOf(variableNames, idStr);
              if (idx == -1) {
                throw new FormatException($"The variable {idStr} occurs in the expression but not in the dataset.");
              }
              expr = Expression.MakeBinary(ExpressionType.ArrayIndex, variableSymbol, Expression.Constant(idx));
            }
            break;
          }
        case TokenEnum.Constant:
          NextSy();
          expr = Expression.Constant(scanner.NumberVal);
          break;
        case TokenEnum.Parameter: {
            expr = MakeParameter(scanner.NumberVal);
            NextSy();
            break;
          }
        case TokenEnum.LeftPar: {
            NextSy();
            expr = Expr();
            if (Sy != TokenEnum.RightPar) throw new FormatException($"expected ')' at position {scanner.Pos}");
            NextSy();
            break;
          }
        default: throw new FormatException($"Unexpected symbol {Sy} " + scanner.ErrorContext);
      }

      // optionally parse power with exponent
      // in Python exponent binds stronger than sign
      // -x**3  = -(x**3)
      if (Sy == TokenEnum.Pow) {
        NextSy();
        expr = Expression.Call(functions["pow"], expr, Fact());
      }
      if (negate) expr = Expression.Negate(expr);
      return expr;
    }


    private Expression MakeParameter(double val) {
      var idx = parameterValues.Count;
      parameterValues.Add(val);
      return Expression.MakeBinary(ExpressionType.ArrayIndex, parameterSymbol, Expression.Constant(idx));
    }

    private Expression GetMethodCall(string idStr, List<Expression> parameters) {
      // we ignore casing
      idStr = idStr.ToLower();
      if (functions.TryGetValue(idStr, out var methodInfo)) {
        return Expression.Call(methodInfo, parameters);
      } else {
        // some functions are translated directly
        switch (idStr) {
          case "sqr": {
              if (parameters.Count != 1) throw new FormatException("sqr() needs exactly one parameter.");
              return Expression.Call(functions["pow"], parameters.First(), Expression.Constant(2.0));
            }
          case "cube": {
              if (parameters.Count != 1) throw new FormatException("cube() needs exactly one parameter.");
              return Expression.Call(functions["pow"], parameters.First(), Expression.Constant(3.0));
            }
          case "logabs": {
              if (parameters.Count != 1) throw new FormatException("logabs() needs exactly one parameter.");
              return Expression.Call(functions["log"], Expression.Call(functions["abs"], parameters.First()));
            }
          case "sqrtabs": {
              if (parameters.Count != 1) throw new FormatException("sqrtabs() needs exactly one parameter.");
              return Expression.Call(functions["sqrt"], Expression.Call(functions["abs"], parameters.First()));
            }
          default: {
              throw new FormatException($"Unknown function {idStr}");
            }
        }
      }
    }

    private List<Expression> ParamList() {
      if (Sy != TokenEnum.LeftPar) throw new FormatException($"Expected '(' for beginning of parameter list at position {scanner.Pos}");
      NextSy();
      var parameters = new List<Expression>();
      parameters.Add(Expr());
      while (Sy == TokenEnum.Comma) {
        NextSy();
        parameters.Add(Expr());
      }

      if (Sy != TokenEnum.RightPar) throw new FormatException($"Expected ')' for end of parameter list " + scanner.ErrorContext);
      NextSy();
      return parameters;
    }

    #region supported functions
    private Dictionary<string, MethodInfo> functions = new Dictionary<string, MethodInfo>() {
      { "abs", typeof(Math).GetMethod("Abs", new Type[] { typeof(double) }) },
      { "sin", typeof(Math).GetMethod("Sin", new Type[] { typeof(double) }) },
      { "cos", typeof(Math).GetMethod("Cos", new Type[] { typeof(double) }) },
      { "tan", typeof(Math).GetMethod("Tan", new Type[] { typeof(double) }) },
      { "tanh", typeof(Math).GetMethod("Tanh", new Type[] { typeof(double) }) },
      { "exp", typeof(Math).GetMethod("Exp", new Type[] { typeof(double) }) },
      { "log", typeof(Math).GetMethod("Log", new Type[] { typeof(double) }) },
      { "sqrt", typeof(Math).GetMethod("Sqrt", new Type[] { typeof(double) }) },
      { "cbrt", typeof(Functions).GetMethod("Cbrt", new Type[] { typeof(double) }) },
      { "pow", typeof(Math).GetMethod("Pow", new Type[] { typeof(double), typeof(double) })},
      { "aq", typeof(Functions).GetMethod("AQ", new Type[] { typeof(double), typeof(double) })},

      { "acos", typeof(Math).GetMethod("Acos", new Type[] { typeof(double) }) },
      { "asin", typeof(Math).GetMethod("Asin", new Type[] { typeof(double) }) },
      { "atan", typeof(Math).GetMethod("Atan", new Type[] { typeof(double) }) },
      { "cosh", typeof(Math).GetMethod("Cosh", new Type[] { typeof(double) }) },
      { "sinh", typeof(Math).GetMethod("Sinh", new Type[] { typeof(double) }) },
      { "ceiling", typeof(Math).GetMethod("Ceiling", new Type[] { typeof(double) }) },
      { "floor", typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }) },
      { "log10", typeof(Math).GetMethod("Log10", new Type[] { typeof(double) }) },
      { "sign", typeof(Math).GetMethod("Sign", new Type[] { typeof(double) }) },

      #endregion
    };

    public class Scanner {
      private const char eofCh = (char)0;
      private readonly string input;
      public string IdStr { get; private set; }
      public double NumberVal { get; private set; }
      public int Pos { get; private set; }
      private char ch;
      public TokenEnum Token { get; private set; }
      public string ErrorContext => $" position {Pos}  {input.Substring(Math.Max(0, Pos - 5))}";

      public Scanner(string input) {
        this.input = input;
        this.Pos = 0;
        this.ch = ' ';
        Token = TokenEnum.None;
      }

      private void NextCh() {
        if (Pos < input.Length) {
          ch = input[Pos];
          Pos++;
        } else ch = eofCh;
      }

      public void NextToken() {
        while (ch == ' ') NextCh();
        Token = TokenEnum.None;
        switch (ch) {
          case '+': Token = TokenEnum.Plus; NextCh(); break;
          case '-': Token = TokenEnum.Minus; NextCh(); break;
          case '*': {
              NextCh();
              if (ch == '*') {
                Token = TokenEnum.Pow;
                NextCh();
              } else {
                Token = TokenEnum.Mul;
              }
              break;
            }
          case '/': Token = TokenEnum.Div; NextCh(); break;
          case '^': Token = TokenEnum.Pow; NextCh(); break;
          case '(': Token = TokenEnum.LeftPar; NextCh(); break;
          case ')': Token = TokenEnum.RightPar; NextCh(); break;
          case ',': Token = TokenEnum.Comma; NextCh(); break;
          case '\'':
          case '\"': {
              // parse everything until ' as identifier
              var quoteCh = ch;
              Token = TokenEnum.Ident;
              NextCh(); // skip '
              var sb = new StringBuilder();
              while (ch != quoteCh) {
                sb.Append(ch);
                NextCh();
              }
              NextCh(); // skip '
              IdStr = sb.ToString();
              break;
            }
          case eofCh: {
              Token = TokenEnum.Eof;
              break;
            }
          default: {
              if (char.IsLetter(ch) || ch == '_') {
                // parse identifier
                var sb = new StringBuilder();
                sb.Append(ch);
                NextCh();
                while (char.IsLetter(ch) || char.IsDigit(ch) || ch == '_') {
                  sb.Append(ch);
                  NextCh();
                }
                Token = TokenEnum.Ident;
                IdStr = sb.ToString();
                break;
              } else if (ch == '.' || char.IsDigit(ch)) {
                // parse number or constant
                // C# rules:
                // Real_Literal
                //     : Decimal_Digit Decorated_Decimal_Digit* '.' Decimal_Digit Decorated_Decimal_Digit* Exponent_Part? Real_Type_Suffix?
                //     | '.' Decimal_Digit Decorated_Decimal_Digit* Exponent_Part? Real_Type_Suffix?
                //     | Decimal_Digit Decorated_Decimal_Digit* Exponent_Part Real_Type_Suffix?
                //     | Decimal_Digit Decorated_Decimal_Digit* Real_Type_Suffix
                //     ;
                // 
                // fragment Exponent_Part
                //     : ('e' | 'E') Sign? Decimal_Digit Decorated_Decimal_Digit*
                //     ;
                // 
                // fragment Sign
                //     : '+' | '-'
                //     ;
                // 
                // fragment Real_Type_Suffix
                //     : 'F' | 'f' | 'D' | 'd' | 'M' | 'm'
                //     ;

                // we only support 'f' suffix (for constants (=fixed) ) and do not allow _ in digits (decoration)

                var sb = new StringBuilder();
                sb.Append(ch);
                NextCh();
                while (char.IsDigit(ch)) {
                  sb.Append(ch);
                  NextCh();
                }
                if (ch == '.') {
                  sb.Append(ch);
                  NextCh();
                  // after '.' we need a least one digit
                  if (!char.IsDigit(ch)) {
                    break;
                  }
                  while (char.IsDigit(ch)) {
                    sb.Append(ch);
                    NextCh();
                  }
                }
                // exponent_part?
                if (ch == 'e' || ch == 'E') {
                  sb.Append(ch);
                  NextCh();
                  // optional sign
                  if (ch == '+' || ch == '-') {
                    sb.Append(ch);
                    NextCh();
                  }
                  // exponent part needs at least one digit
                  if (!char.IsDigit(ch)) {
                    break;
                  }
                  while (char.IsDigit(ch)) {
                    sb.Append(ch);
                    NextCh();
                  }
                }

                // optional suffix
                if (ch == 'f') {
                  NextCh();
                  Token = TokenEnum.Constant;
                  NumberVal = double.Parse(sb.ToString());
                } else {
                  Token = TokenEnum.Parameter;
                  NumberVal = double.Parse(sb.ToString());
                }
              } else Token = TokenEnum.None;
              break;
            }
        }
      }
    }

  }
}
