package android.content;
public class ContentValues {
  final java.util.Map<String,Object> values=new java.util.HashMap<>();
  public void put(String k,String v){values.put(k,v);}
  public void put(String k,Integer v){values.put(k,v);}
  public void clear(){values.clear();}
}
