package android.content;
import java.io.*;
import java.nio.file.*;
import android.net.Uri;
// Only the Android storage boundary is mocked. ZIP compression, streaming and
// the production C++ JNI calls run against a real desktop JVM.
public class ContentResolver {
  public Uri insert(Uri collection,ContentValues values) throws IOException {
    if(!Integer.valueOf(1).equals(values.values.get("is_pending")))throw new IOException("Must create pending file");
    if(!"Download/G1Quest/".equals(values.values.get("relative_path")))throw new IOException("Wrong shared directory");
    Path p=Paths.get(System.getProperty("export.root"),(String)values.values.get("_display_name"));
    Files.createDirectories(p.getParent());Files.writeString(Paths.get(p+".pending"),"pending");return new Uri(p.toString());
  }
  public OutputStream openOutputStream(Uri uri) throws IOException {
    if(uri.path.contains("fail"))return new OutputStream(){public void write(int b)throws IOException{throw new IOException("Injected full storage");}};
    return new FileOutputStream(uri.path);
  }
  public int update(Uri uri,ContentValues values,String where,String[] args)throws IOException {
    if(!Integer.valueOf(0).equals(values.values.get("is_pending")))throw new IOException("Must publish completed file");
    // The central directory must already exist before publication.
    try(java.util.zip.ZipFile z=new java.util.zip.ZipFile(uri.path)){if(z.size()<1)throw new IOException("Empty ZIP");}
    Files.delete(Paths.get(uri.path+".pending"));return 1;
  }
  public int delete(Uri uri,String where,String[] args)throws IOException {
    Files.deleteIfExists(Paths.get(uri.path));Files.deleteIfExists(Paths.get(uri.path+".pending"));return 1;
  }
}
